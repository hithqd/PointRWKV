import yaml
from easydict import EasyDict


class EasyConfig(EasyDict):
    """Extended EasyDict that supports YAML loading."""
    
    def load(self, path, recursive=True):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if data is not None:
            self.update(data)
        return self


def cfg_from_yaml_file(cfg_file):
    """Load config from a YAML file."""
    config = EasyConfig()
    config.load(cfg_file)
    return config

