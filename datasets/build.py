from utils.registry import Registry


DATASETS = Registry('dataset')


def build_dataset_from_cfg(cfg, default_args=None):
    """Build a dataset from config."""
    dataset_name = cfg.get('NAME', None) or cfg.get('type', None)
    if dataset_name is None:
        raise KeyError('cfg must contain key "NAME" or "type"')
    
    dataset_cls = DATASETS.get(dataset_name)
    if dataset_cls is None:
        raise KeyError(f'{dataset_name} is not registered. Available: {DATASETS.module_dict.keys()}')
    
    return dataset_cls(cfg)
