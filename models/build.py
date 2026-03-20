"""Model builder for PointRWKV."""

from models.point_rwkv_cls import PointRWKVCls
from models.point_rwkv_seg import PointRWKVPartSeg
from models.point_rwkv_pretrain import PointRWKVPretrain


MODEL_REGISTRY = {
    'PointRWKVCls': PointRWKVCls,
    'PointRWKVPartSeg': PointRWKVPartSeg,
    'PointRWKVPretrain': PointRWKVPretrain,
}


def build_model(config):
    """Build a model from config.
    
    Args:
        config: EasyConfig with 'model_name' key
    Returns:
        model: nn.Module
    """
    model_name = config.get('model_name', 'PointRWKVCls')
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}')
    
    model = MODEL_REGISTRY[model_name](config)
    return model
