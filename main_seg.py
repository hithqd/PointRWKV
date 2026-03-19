"""PointRWKV Part Segmentation Training & Testing Script.

Supports ShapeNetPart segmentation.

Usage:
    # Train with pre-trained backbone
    python main_seg.py --config cfgs/seg_shapenetpart.yaml --exp_name seg_snp \
        --ckpt experiments/pretrain/ckpts/epoch_300.pth
    
    # Test
    python main_seg.py --config cfgs/seg_shapenetpart.yaml --test \
        --ckpt experiments/seg_snp/ckpts/best.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict

from utils.config import cfg_from_yaml_file
from utils.logger import get_logger, print_log
from utils.misc import set_random_seed, AverageMeter, worker_init_fn
from utils.checkpoint import save_checkpoint, load_checkpoint, load_model_weights
from models.point_rwkv_seg import PointRWKVPartSeg
from datasets import build_dataset_from_cfg
from datasets.ShapeNetPartDataset import ShapeNetPart


def parse_args():
    parser = argparse.ArgumentParser('PointRWKV Part Segmentation')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--exp_name', type=str, default='seg', help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', type=int, default=8, help='data loading workers')
    parser.add_argument('--ckpt', type=str, default=None, help='pretrained/resume checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--test', action='store_true', help='test mode')
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


def build_seg_dataloader(config, args, split='train'):
    dataset_config = config.dataset.copy()
    dataset_config.subset = split
    dataset = build_dataset_from_cfg(dataset_config)
    
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == 'train'))
    else:
        sampler = None
    
    loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=(split == 'train' and sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        worker_init_fn=worker_init_fn,
        collate_fn=collate_seg_batch,
    )
    return loader, sampler, dataset


def collate_seg_batch(batch):
    """Custom collate for segmentation data."""
    points_list, cls_list, seg_list, cat_list = [], [], [], []
    for item in batch:
        _, _, data = item
        points_list.append(data[0])
        cls_list.append(data[1])
        seg_list.append(data[2])
        cat_list.append(data[3])
    
    points = torch.stack(points_list)
    cls_label = torch.stack(cls_list)
    seg_label = torch.stack(seg_list)
    cat_idx = torch.tensor(cat_list)
    
    return points, cls_label, seg_label, cat_idx


def train_one_epoch(model, train_loader, optimizer, device, epoch, epochs, logger):
    model.train()
    loss_meter = AverageMeter()
    
    for i, batch in enumerate(train_loader):
        points, cls_label, seg_label, cat_idx = batch
        points = points.to(device)
        cls_label = cls_label.to(device)
        seg_label = seg_label.to(device)
        
        ret = model(points, cls_label, seg_label)
        loss = ret['loss']
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        loss_meter.update(loss.item(), points.shape[0])
        
        if i % 50 == 0:
            print_log(
                f'Epoch [{epoch}/{epochs}] Iter [{i}/{len(train_loader)}] '
                f'Loss: {loss_meter.avg:.4f}',
                logger=logger
            )
    
    return loss_meter.avg


@torch.no_grad()
def validate_seg(model, test_loader, device, logger, seg_classes):
    model.eval()
    
    all_shape_ious = defaultdict(list)
    
    for batch in test_loader:
        points, cls_label, seg_label, cat_idx = batch
        points = points.to(device)
        cls_label = cls_label.to(device)
        
        ret = model(points, cls_label)
        logits = ret['logits']  # (B, N, num_parts)
        
        pred = logits.argmax(dim=-1).cpu().numpy()
        target = seg_label.numpy()
        cat_idx_np = cat_idx.numpy()
        
        # Compute IoU per shape
        cat_names = list(seg_classes.keys())
        
        for b in range(pred.shape[0]):
            cat_name = cat_names[cat_idx_np[b]]
            part_ids = seg_classes[cat_name]
            
            part_ious = []
            for part in part_ids:
                I = np.sum(np.logical_and(pred[b] == part, target[b] == part))
                U = np.sum(np.logical_or(pred[b] == part, target[b] == part))
                if U == 0:
                    iou = 1.0
                else:
                    iou = I / U
                part_ious.append(iou)
            
            all_shape_ious[cat_name].append(np.mean(part_ious))
    
    # Compute mean IoU
    all_cat_ious = {}
    for cat_name in all_shape_ious:
        all_cat_ious[cat_name] = np.mean(all_shape_ious[cat_name])
    
    mean_shape_iou = np.mean([np.mean(v) for v in all_shape_ious.values()])
    
    # Instance mIoU
    all_instance_ious = []
    for v in all_shape_ious.values():
        all_instance_ious.extend(v)
    instance_miou = np.mean(all_instance_ious)
    
    print_log(f'Instance mIoU: {instance_miou*100:.2f}%', logger=logger)
    print_log(f'Category mIoU: {mean_shape_iou*100:.2f}%', logger=logger)
    for cat_name, iou in sorted(all_cat_ious.items()):
        print_log(f'  {cat_name}: {iou*100:.2f}%', logger=logger)
    
    return instance_miou, mean_shape_iou


def main():
    args = parse_args()
    config = cfg_from_yaml_file(args.config)
    
    init_distributed(args)
    set_random_seed(args.seed + (args.rank if hasattr(args, 'rank') else 0))
    
    exp_dir = os.path.join('experiments', args.exp_name)
    if not args.distributed or args.rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'ckpts'), exist_ok=True)
    
    log_file = os.path.join(exp_dir, 'seg.log') if (not args.distributed or args.rank == 0) else None
    logger = get_logger('seg', log_file=log_file)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = PointRWKVPartSeg(config).to(device)
    
    # Load pre-trained weights
    if args.ckpt and not args.resume and not args.test:
        load_model_weights(model.backbone, args.ckpt, logger=logger)
        print_log(f'Loaded pre-trained backbone from {args.ckpt}', logger=logger)
    
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # Seg classes for evaluation
    seg_classes = ShapeNetPart.seg_classes
    
    # Test mode
    if args.test:
        if args.ckpt:
            load_checkpoint(model.module if args.distributed else model, args.ckpt, logger=logger)
        test_loader, _, _ = build_seg_dataloader(config, args, split='test')
        validate_seg(model.module if args.distributed else model, test_loader, device, logger, seg_classes)
        return
    
    # Train mode
    train_loader, train_sampler, _ = build_seg_dataloader(config, args, split='train')
    test_loader, _, _ = build_seg_dataloader(config, args, split='test')
    
    # Optimizer
    lr = config.get('lr', 5e-4)
    weight_decay = config.get('weight_decay', 0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
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
    
    # Training
    best_iou = 0.0
    best_epoch = 0
    
    for epoch in range(start_epoch, epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, epochs, logger)
        scheduler.step()
        
        if not args.distributed or args.rank == 0:
            if (epoch + 1) % config.get('val_freq', 1) == 0:
                inst_iou, cat_iou = validate_seg(
                    model.module if args.distributed else model,
                    test_loader, device, logger, seg_classes
                )
                
                if inst_iou > best_iou:
                    best_iou = inst_iou
                    best_epoch = epoch + 1
                    save_checkpoint(
                        model.module if args.distributed else model,
                        optimizer, epoch + 1,
                        os.path.join(exp_dir, 'ckpts', 'best.pth'),
                        logger=logger
                    )
                
                print_log(
                    f'Epoch [{epoch+1}/{epochs}] Best mIoU: {best_iou*100:.2f}% (Epoch {best_epoch})',
                    logger=logger
                )
            
            if (epoch + 1) % config.get('save_freq', 50) == 0:
                save_checkpoint(
                    model.module if args.distributed else model,
                    optimizer, epoch + 1,
                    os.path.join(exp_dir, 'ckpts', f'epoch_{epoch+1}.pth'),
                    logger=logger
                )
    
    if not args.distributed or args.rank == 0:
        print_log(f'Training completed! Best mIoU: {best_iou*100:.2f}% at Epoch {best_epoch}', logger=logger)


if __name__ == '__main__':
    main()
