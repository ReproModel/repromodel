def enforce_types_and_ranges(config):
    def decorator(func):
        from functools import wraps
        import inspect

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Retrieve the function signature and bind the passed arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Create a new arguments dictionary from the bound arguments
            all_args = bound_args.arguments

            # Validate arguments based on the config specifications
            for param, specs in config.items():
                value = all_args.get(param)

                if 'type' in specs and not isinstance(value, specs['type']) and value is not None:
                    raise TypeError(f"{param} must be of type {specs['type'].__name__}, got type {type(value).__name__}")

                if 'range' in specs and value is not None:
                    if not (specs['range'][0] <= value <= specs['range'][1]):
                        raise ValueError(f"{param} must be between {specs['range'][0]} and {specs['range'][1]}")

                if 'options' in specs and value not in specs['options'] and value is not None:
                    raise ValueError(f"{param} must be one of {specs['options']}")

            # Now call the original function with the correct arguments
            return func(**all_args)

        return wrapper
    return decorator


def tag(**tags):
    def decorator(cls):
        # Initialize default values if not provided
        default_tags = {
            'task': [""],
            'subtask': [""],
            'modality': [""],
            'submodality': [""]
        }
        
        # Update default values with provided tags
        default_tags.update(tags)
        
        # Set attributes on the class
        for key, value in default_tags.items():
            setattr(cls, key, value)
            
        return cls
    return decorator