import os
import torch
from utils.logger import print_log


def save_checkpoint(model, optimizer, epoch, path, logger=None):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, path)
    print_log(f'Saved checkpoint to {path}', logger=logger)


def load_checkpoint(model, path, optimizer=None, logger=None, strict=True):
    """Load checkpoint."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'No checkpoint found at {path}')
    
    checkpoint = torch.load(path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    else:
        state_dict = checkpoint

    # Handle DataParallel/DistributedDataParallel wrapper
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=strict)
    print_log(f'Loaded checkpoint from {path}', logger=logger)
    if msg.missing_keys:
        print_log(f'Missing keys: {msg.missing_keys}', logger=logger)
    if msg.unexpected_keys:
        print_log(f'Unexpected keys: {msg.unexpected_keys}', logger=logger)

    epoch = checkpoint.get('epoch', 0)
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return epoch



def load_model_weights(model, path, logger=None):
    """Load pre-trained weights with key mapping from MAE to backbone.
    
    MAE encoder keys map to PointRWKV backbone keys:
        encoder_blocks.{i}.* -> blocks.0.{i}.*
        encoder_embed.*      -> embed_modules.0.*
        group.*              -> group_modules.0.*
        encoder_pos.*        -> pos_embed.0.*
        encoder_norm.*       -> norms.0.*
    """
    if not os.path.isfile(path):
        print_log(f'No checkpoint found at {path}, skip loading', logger=logger)
        return
    
    checkpoint = torch.load(path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    else:
        state_dict = checkpoint
    
    # Strip 'module.' prefix if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        clean_state_dict[k] = v
    state_dict = clean_state_dict
    
    # Try direct loading first
    msg = model.load_state_dict(state_dict, strict=False)
    loaded_keys = set(state_dict.keys()) - set(msg.unexpected_keys)
    
    if len(loaded_keys) > 0:
        print_log(f'Loaded {len(loaded_keys)} keys directly from {path}', logger=logger)
        if msg.missing_keys:
            print_log(f'Missing keys ({len(msg.missing_keys)}): {msg.missing_keys[:5]}...', logger=logger)
        return
    
    # If direct loading failed, try MAE -> backbone key mapping
    print_log('Direct loading found no matching keys, trying MAE key mapping...', logger=logger)
    
    mapped_state_dict = {}
    for k, v in state_dict.items():
        new_key = None
        
        # encoder_blocks.{i}.* -> blocks.0.{i}.*
        if k.startswith('encoder_blocks.'):
            new_key = k.replace('encoder_blocks.', 'blocks.0.', 1)
        # encoder_embed.* -> embed_modules.0.*
        elif k.startswith('encoder_embed.'):
            new_key = k.replace('encoder_embed.', 'embed_modules.0.', 1)
        # group.* -> group_modules.0.*
        elif k.startswith('group.'):
            new_key = k.replace('group.', 'group_modules.0.', 1)
        # encoder_pos.* -> pos_embed.0.*
        elif k.startswith('encoder_pos.'):
            new_key = k.replace('encoder_pos.', 'pos_embed.0.', 1)
        # encoder_norm.* -> norms.0.*
        elif k.startswith('encoder_norm.'):
            new_key = k.replace('encoder_norm.', 'norms.0.', 1)
        
        if new_key is not None:
            mapped_state_dict[new_key] = v
    
    if len(mapped_state_dict) > 0:
        msg = model.load_state_dict(mapped_state_dict, strict=False)
        loaded = set(mapped_state_dict.keys()) - set(msg.unexpected_keys)
        print_log(f'Loaded {len(loaded)} mapped keys from MAE checkpoint', logger=logger)
        if msg.missing_keys:
            print_log(f'Remaining missing keys ({len(msg.missing_keys)})', logger=logger)
    else:
        print_log(f'Warning: No keys could be mapped from {path}', logger=logger)
