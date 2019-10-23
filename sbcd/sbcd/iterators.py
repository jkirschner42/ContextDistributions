

def grid_search_delta():
    config = []
    betas = [0.01, 0.05, 0.1, 0.15, 0.2, 0,0.3, 0.5, 1 ,2 ]
    # betas = [0.05,  0.075, 0.1, 0.15, 0.2]
    betas_ucb = [0.01, 0.05, 0.1, 0.5, 1, 2]
    # betas_ucb = [0.01, 0.05, 0.1, 0.5, 1]
    config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'DIRECT'})
    config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'UCB'})
    config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'DIRECT'})
    config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'UCB'})
    config += _get_grid_search_config('febo.algorithms.ucb.UCB', betas_ucb)

    return config


def grid_search_env1():
    config = []
    betas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2, 5]
    # betas = [0.05, 0.1, 0.2, 1, 2]
    betas_ucb = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    # betas_ucb = [0.01, 0.05, 0.1, 0.5, 1]
    config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'DIRECT'})
    # config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'UCB'})

    # config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'DIRECT',
    #                                                                    'algorithm.ids:use_1pa': True})

    config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method': 'DIRECT',
                                                                       'algorithm.ids:use_1pa': False})

    # config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'UCB'})
    config += _get_grid_search_config('febo.algorithms.ucb.UCB', betas_ucb)

    return config

def grid_search_agroscope():
    config = []
    betas = [0.1, 0.2, 0.5, 1, 2.5,  5, 8, 10, 20]
    betas = [1, 2.5, 5, 7, 8, 10, 20, 30]
    betas = [10, 11, 12, 15, 20, 30]
    # betas = [5, 6.5, 8, 10, 15, 20, 30, 50]
    # betas = [10, 15, 20]
    # betas = [0.05,  0.075, 0.1, 0.15, 0.2]
    betas_ucb = [1, 2.5, 5, 7, 8, 10, 20, 30]
    # betas_ucb = [5, 6.5, 8, 10, 15, 20, 30, 50]
    # betas_ucb = [10, 15, 20]
    # betas_ucb = [0.01, 0.05, 0.1, 0.5, 1]
    # config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'DIRECT'})
    # config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'UCB'})
    # config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'DIRECT',
    #                                                                    'algorithm.ids:use_1pa' : True })
    config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'DIRECT',
                                                                       'algorithm.ids:use_1pa' : False })
    # config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'UCB'})
    # config += _get_grid_search_config('febo.algorithms.ucb.UCB', betas_ucb)

    return config


def grid_search_smab():
    config = []
    betas = [0.1, 0.2, 0.5, 1, 2.5,  5, 8, 10, 20]
    betas_ucb = [0.1, 1, 2.5, 5, 7, 8, 10, 20]
    # config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'DIRECT'})
    # config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'UCB'})
    config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'DIRECT'})
                                                                       # 'algorithm.ids:use_1pa' : True })
    # config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'DIRECT',
    #                                                                    'algorithm.ids:use_1pa' : False })
    # config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'UCB'})
    config += _get_grid_search_config('febo.algorithms.ucb.UCB', betas_ucb)

    return config

def grid_search_amab():
    config = []
    betas = [5, 6, 7, 8, 10]
    # betas = [5,6,7,11]
    # betas = [7.5,8,8.5]
    # betas = [8, 20]
    # config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'DIRECT'})
    # config += _get_grid_search_config('sids.algorithms.Bose', betas, {'algorithm.bose:compute_method' : 'UCB'})
    # config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'DIRECT',
    #                                                                    'algorithm.ids:use_1pa' : False })
    config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'DIRECT',
                                                                       'algorithm.ids:use_1pa' : False })
    # config += _get_grid_search_config('sids.algorithms.IDS_S', betas, {'algorithm.ids:compute_method' : 'UCB'})
    config += _get_grid_search_config('febo.algorithms.ucb.UCB', betas)

    return config



def _get_grid_search_config(algorithm, betas, additional_config={}):
    config = []
    for beta in betas:
        algo_config = {'experiment.simple:algorithm': algorithm,
                       'model:beta': beta}
        algo_config.update(additional_config)
        config.append(algo_config)
    return config