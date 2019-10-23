def label(id, config):
    if config['environment.benchmark']['exact_context']:
        return f'exact'

    l = config['algorithm.ucbcd']['l']

    if config['algorithm.ucbcd']['observe_context']:
        return f'observed-{l}'
    else:
        return f'hidden-{l}'


def beta(id, config):
    return f"beta={config['model']['beta']}"
