import numpy as np

def log_t(model):
    return np.max([2., np.log(model.t + 1.)])

def sqrt_log_t(model):
    return np.sqrt(np.max([2., np.log(model.t + 1.)]))

def logdet(model):
    logdet = model._get_logdet()
    logdet_priornoise = model._get_logdet_prior_noise()
    return np.asscalar(np.sqrt(2*np.log(1 / model.delta) + (logdet - logdet_priornoise))/4 + model._norm()/4)
    # return model.domain.d*np.log(model.domain.d)

def linear(model):
    return np.sqrt(model.domain.d*np.log(model.t + 1) + 2*np.log(1/model.delta)) + 1
