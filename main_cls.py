"""PointRWKV Classification Training & Testing Script.

Supports:
- Training from scratch
- Fine-tuning from pre-trained checkpoint
- Testing with voting

Usage:
    # Train from scratch on ModelNet40
    python main_cls.py --config cfgs/cls_modelnet40.yaml --exp_name cls_mn40
    
    # Fine-tune from pre-trained model
    python main_cls.py --config cfgs/cls_modelnet40.yaml --exp_name cls_mn40_ft \
        --ckpt experiments/pretrain/ckpts/epoch_300.pth
    
    # Test
    python main_cls.py --config cfgs/cls_modelnet40.yaml --test --ckpt /path/to/ckpt \
        --vote
"""

import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.config import cfg_from_yaml_file
from utils.logger import get_logger, print_log
from utils.misc import set_random_seed, AverageMeter, worker_init_fn
from utils.checkpoint import save_checkpoint, load_checkpoint, load_model_weights
from models.point_rwkv_cls import PointRWKVCls
from datasets import build_dataset_from_cfg
from datasets.data_transforms import PointcloudScaleAndTranslate


def parse_args():
    parser = argparse.ArgumentParser('PointRWKV Classification')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--exp_name', type=str, default='cls', help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', type=int, default=8, help='data loading workers')
    parser.add_argument('--ckpt', type=str, default=None, help='pretrained/resume checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training (load optimizer state)')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--vote', action='store_true', help='voting during test')
    parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch', 'slurm'])
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    return args


def init_distributed(args):
    if args.launcher == 'none':
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        return
    
    args.distributed = True
    if args.launcher == 'pytorch':
        args.rank = int(os.environ.get('RANK', 0))
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        args.gpu = int(os.environ.get('LOCAL_RANK', 0))
    elif args.launcher == 'slurm':
        args.rank = int(os.environ.get('SLURM_PROCID', 0))
        args.world_size = int(os.environ.get('SLURM_NTASKS', 1))
        args.gpu = args.rank % torch.cuda.device_count()
    
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=args.rank)


def build_cls_dataloader(config, args, split='train'):
    dataset_config = config.dataset.copy()
    dataset_config.subset = split
    dataset = build_dataset_from_cfg(dataset_config)
    
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == 'train'))
    else:
        sampler = None
    
    loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=(split == 'train' and sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        worker_init_fn=worker_init_fn,
    )
    return loader, sampler


def extract_batch_data(batch):
    """Extract points and labels from dataset return format.
    
    After DataLoader collation, the batch format is:
        (list_of_strs, list_of_strs, (stacked_points, stacked_labels))
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        # Check if first element is a list of strings (post-collation)
        if isinstance(batch[0], (list, tuple)) and len(batch[0]) > 0 and isinstance(batch[0][0], str):
            _, _, data = batch
            points, label = data[0], data[1]
        elif isinstance(batch[0], str):
            # Single sample (unlikely with DataLoader)
            _, _, data = batch
            points, label = data[0], data[1]
        else:
            points, label = batch[0], batch[1]
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        points, label = batch[0], batch[1]
    else:
        raise ValueError(f'Unknown batch format: {type(batch)}, len={len(batch)}')
    
    return points, label


def train_one_epoch(model, train_loader, optimizer, device, epoch, epochs, logger, 
                    train_transforms=None):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for i, batch in enumerate(train_loader):
        points, label = extract_batch_data(batch)
        points = points.to(device)
        label = label.to(device).long()
        
        # Data augmentation (on GPU)
        if train_transforms is not None:
            points = train_transforms(points)
        
        ret = model(points, label)
        loss = ret['loss']
        logits = ret['logits']
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Compute accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == label).float().mean()
        
        loss_meter.update(loss.item(), points.shape[0])
        acc_meter.update(acc.item(), points.shape[0])
        
        if i % 50 == 0:
            print_log(
                f'Epoch [{epoch}/{epochs}] Iter [{i}/{len(train_loader)}] '
                f'Loss: {loss_meter.avg:.4f} Acc: {acc_meter.avg:.4f}',
                logger=logger
            )
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, test_loader, device, logger, num_votes=1):
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    count = 0
    
    for batch in test_loader:
        points, label = extract_batch_data(batch)
        points = points.to(device)
        label = label.to(device).long()
        
        if num_votes > 1:
            # Voting
            all_logits = []
            for v in range(num_votes):
                scale = torch.FloatTensor(1).uniform_(0.9, 1.1).to(device)
                pts_aug = points * scale
                ret = model(pts_aug)
                all_logits.append(ret['logits'])
            logits = torch.stack(all_logits, dim=0).mean(dim=0)
        else:
            ret = model(points, label)
            logits = ret['logits']
            total_loss += ret['loss'].item() * points.shape[0]
        
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(label.cpu())
        count += points.shape[0]
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).float().mean().item()
    avg_loss = total_loss / count if count > 0 else 0
    
    # Per-class accuracy
    classes = torch.unique(all_labels)
    class_accs = []
    for c in classes:
        mask = all_labels == c
        class_acc = (all_preds[mask] == all_labels[mask]).float().mean().item()
        class_accs.append(class_acc)
    mean_class_acc = np.mean(class_accs)
    
    print_log(
        f'Test: Overall Acc: {acc*100:.2f}% | Mean Class Acc: {mean_class_acc*100:.2f}%',
        logger=logger
    )
    
    return acc, mean_class_acc, avg_loss


def main():
    args = parse_args()
    config = cfg_from_yaml_file(args.config)
    
    init_distributed(args)
    set_random_seed(args.seed + (args.rank if hasattr(args, 'rank') else 0))
    
    # Setup directories
    exp_dir = os.path.join('experiments', args.exp_name)
    if not args.distributed or args.rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'ckpts'), exist_ok=True)
    
    log_file = os.path.join(exp_dir, 'cls.log') if (not args.distributed or args.rank == 0) else None
    logger = get_logger('cls', log_file=log_file)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = PointRWKVCls(config).to(device)
    
    # Load pre-trained weights (if provided and not resuming)
    if args.ckpt and not args.resume:
        load_model_weights(model.backbone, args.ckpt, logger=logger)
        print_log(f'Loaded pre-trained backbone from {args.ckpt}', logger=logger)
    
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # Test mode
    if args.test:
        if args.ckpt:
            load_checkpoint(model.module if args.distributed else model, args.ckpt, logger=logger)
        test_loader, _ = build_cls_dataloader(config, args, split='test')
        num_votes = 10 if args.vote else 1
        validate(model.module if args.distributed else model, test_loader, device, logger, num_votes)
        return
    
    # Train mode
    train_loader, train_sampler = build_cls_dataloader(config, args, split='train')
    test_loader, _ = build_cls_dataloader(config, args, split='test')
    
    # Optimizer
    lr = config.get('lr', 3e-4)
    weight_decay = config.get('weight_decay', 0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # Scheduler
    epochs = config.get('epochs', 300)
    warmup_epochs = config.get('warmup_epochs', 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(epoch / warmup_epochs, 1e-6)
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume
    start_epoch = 0
    if args.resume and args.ckpt:
        start_epoch = load_checkpoint(
            model.module if args.distributed else model,
            args.ckpt, optimizer=optimizer, logger=logger
        )
    
    # Data augmentation
    train_transforms = PointcloudScaleAndTranslate()
    
    # Training
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(start_epoch, epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, epochs, logger, train_transforms
        )
        scheduler.step()
        
        if not args.distributed or args.rank == 0:
            # Validate
            if (epoch + 1) % config.get('val_freq', 1) == 0:
                acc, _, val_loss = validate(
                    model.module if args.distributed else model,
                    test_loader, device, logger
                )
                
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
                    save_checkpoint(
                        model.module if args.distributed else model,
                        optimizer, epoch + 1,
                        os.path.join(exp_dir, 'ckpts', 'best.pth'),
                        logger=logger
                    )
                
                print_log(
                    f'Epoch [{epoch+1}/{epochs}] Best Acc: {best_acc*100:.2f}% (Epoch {best_epoch})',
                    logger=logger
                )
            
            # Save periodic checkpoint
            if (epoch + 1) % config.get('save_freq', 50) == 0:
                save_checkpoint(
                    model.module if args.distributed else model,
                    optimizer, epoch + 1,
                    os.path.join(exp_dir, 'ckpts', f'epoch_{epoch+1}.pth'),
                    logger=logger
                )
    
    if not args.distributed or args.rank == 0:
        print_log(f'Training completed! Best Acc: {best_acc*100:.2f}% at Epoch {best_epoch}', logger=logger)


if __name__ == '__main__':
    main()
