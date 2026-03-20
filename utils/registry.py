class Registry:
    """A registry to map strings to classes."""

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self._name}, items={list(self._module_dict.keys())})'

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def register(self, module_class):
        """Register a module class.
        
        Can be used as a decorator:
            @registry.register
            class MyClass:
                pass
        """
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self._name}')
        self._module_dict[module_name] = module_class
        return module_class

    def register_module(self):
        """Register a module class (decorator with parentheses).
        
        Can be used as:
            @registry.register_module()
            class MyClass:
                pass
        """
        def decorator(module_class):
            module_name = module_class.__name__
            if module_name in self._module_dict:
                raise KeyError(f'{module_name} is already registered in {self._name}')
            self._module_dict[module_name] = module_class
            return module_class
        return decorator

    def build(self, cfg):
        """Build a module from config dict."""
        obj_type = cfg.get('NAME', None) or cfg.get('type', None)
        if obj_type is None:
            raise KeyError(f'`cfg` must contain key "NAME" or "type", but got {cfg}')
        if obj_type not in self._module_dict:
            raise KeyError(f'{obj_type} is not in the {self._name} registry. '
                           f'Available: {list(self._module_dict.keys())}')
        obj_cls = self._module_dict[obj_type]
        return obj_cls(cfg)
