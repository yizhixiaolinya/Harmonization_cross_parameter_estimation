import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):

    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']

    model = models[model_spec['name']](**model_args)

    if load_sd:
        for key in ['sd_P', 'sd_G', 'sd_D', 'state_dict']:
            if key in model_spec:
                model.load_state_dict(model_spec[key])
                break
        else:
            raise KeyError("No valid state_dict key found in model_spec (tried: 'sd_P', 'sd_G', 'sd_D', 'state_dict')")

    return model