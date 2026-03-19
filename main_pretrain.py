"""PointRWKV Pre-training Script.

Pre-trains the PointRWKV backbone using Masked Autoencoding on ShapeNet.

Usage:
    python main_pretrain.py --config cfgs/pretrain.yaml
    
    # Multi-GPU
    python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
        --config cfgs/pretrain.yaml --launcher pytorch
"""

import os
import argparse
import time
import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.config import cfg_from_yaml_file
from utils.logger import get_logger, print_log
from utils.misc import set_random_seed, AverageMeter, worker_init_fn
from utils.checkpoint import save_checkpoint, load_checkpoint
from models.point_rwkv_pretrain import PointRWKVPretrain
from datasets import build_dataset_from_cfg


def parse_args():
    parser = argparse.ArgumentParser('PointRWKV Pre-training')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--exp_name', type=str, default='pretrain', help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', type=int, default=8, help='data loading workers')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint path')
    parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch', 'slurm'])
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    return args


def init_distributed(args):
    """Initialize distributed training."""
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
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )


def main():
    args = parse_args()
    config = cfg_from_yaml_file(args.config)
    
    # Distributed init
    init_distributed(args)
    
    # Setup
    set_random_seed(args.seed + args.rank if hasattr(args, 'rank') else args.seed)
    
    # Create experiment directory
    exp_dir = os.path.join('experiments', args.exp_name)
    if not args.distributed or args.rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'ckpts'), exist_ok=True)
    
    # Logger
    log_file = os.path.join(exp_dir, 'pretrain.log') if (not args.distributed or args.rank == 0) else None
    logger = get_logger('pretrain', log_file=log_file)
    
    if not args.distributed or args.rank == 0:
        print_log(f'Config: {config}', logger=logger)
    
    # Dataset
    dataset_config = config.dataset
    dataset_config.subset = 'train'
    train_dataset = build_dataset_from_cfg(dataset_config)
    
    if args.distributed:
        sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    
    # Model
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = PointRWKVPretrain(config).to(device)
    
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # Optimizer
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # Scheduler: Cosine with warmup
    epochs = config.get('epochs', 300)
    warmup_epochs = config.get('warmup_epochs', 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(epoch / warmup_epochs, 1e-6)
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            model.module if args.distributed else model,
            args.resume, optimizer=optimizer, logger=logger
        )
    
    # Training loop
    print_log(f'Start pre-training from epoch {start_epoch}', logger=logger)
    
    for epoch in range(start_epoch, epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        model.train()
        
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            # After DataLoader collation, batch is:
            # (list_of_taxonomy_ids, list_of_model_ids, stacked_point_tensor)
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                pts = batch[-1]  # The last element is the stacked point tensor
            elif isinstance(batch, (list, tuple)) and len(batch) == 1:
                pts = batch[0]
            else:
                pts = batch
            
            if isinstance(pts, (list, tuple)):
                pts = pts[0]  # Unwrap if still wrapped
            
            pts = pts.to(device)
            
            loss = model(pts)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            loss_meter.update(loss.item(), pts.shape[0])
            
            if i % 50 == 0 and (not args.distributed or args.rank == 0):
                print_log(
                    f'Epoch [{epoch}/{epochs}] Iter [{i}/{len(train_loader)}] '
                    f'Loss: {loss_meter.avg:.4f} LR: {optimizer.param_groups[0]["lr"]:.6f}',
                    logger=logger
                )
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        if not args.distributed or args.rank == 0:
            print_log(
                f'Epoch [{epoch}/{epochs}] Loss: {loss_meter.avg:.4f} '
                f'Time: {datetime.timedelta(seconds=int(epoch_time))}',
                logger=logger
            )
            
            # Save checkpoint
            if (epoch + 1) % config.get('save_freq', 50) == 0 or epoch == epochs - 1:
                save_checkpoint(
                    model.module if args.distributed else model,
                    optimizer, epoch + 1,
                    os.path.join(exp_dir, 'ckpts', f'epoch_{epoch+1}.pth'),
                    logger=logger
                )
    
    if not args.distributed or args.rank == 0:
        print_log('Pre-training completed!', logger=logger)


if __name__ == '__main__':
    main()
