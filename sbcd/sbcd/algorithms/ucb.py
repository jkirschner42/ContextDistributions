from febo.algorithms import Algorithm, AlgorithmConfig
from febo.utils.config import ConfigField, assign_config
from febo.models.lls import LinearLeastSquares

import numpy as np


class UCBCDConfig(AlgorithmConfig):
    observe_context = ConfigField(False, comment='If true, exact context is used for regression')
    l = ConfigField(1)
    _section = 'algorithm.ucbcd'

class LLS(LinearLeastSquares):
    def __init__(self, domain, sigma):
        self.sigma = sigma
        super().__init__(domain)

    def _update_cached(self):
        # compute beta with noise scaling
        self._theta = np.linalg.solve(self._V, self._Y)
        self._detV = np.linalg.det(self.V)
        self._beta_t = (self.sigma*np.sqrt(np.log(self.detV) - 2 * np.log(self.delta)) + 1)

@assign_config(UCBCDConfig)
class UCBCD(Algorithm):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.K = self.domain.num_points # number of actions
        self.d = self.domain.d # feature dimension

        try:
            self.noise_var = np.asscalar(kwargs['noise_function'](None))**2
        except KeyError:
            self.noise_var = 0.01
            print("setting noise var to 0.01")

        # compute subgaussian proxy
        if self.config.observe_context:
            sigma = np.sqrt(self.noise_var)
        else:
            sigma = np.sqrt(4 + self.noise_var)

        self.m = LLS(domain=self.domain, sigma=sigma)  # linear model

    def _next(self, context):
        sampler = context['sampler']

        # compute feature set
        Phi = sampler(self.config.l)

        # compute ucb score:
        x_ucb = np.argmax(self.m.ucb(Phi))
        h_ee_ucb = Phi[x_ucb]

        return x_ucb, {'h_ee': h_ee_ucb}

    def _get_dtype_fields(self):
        fields = super()._get_dtype_fields()
        return fields + [('h_ee', f'({self.domain.d},)f8')]  # add field for expected feature

    def add_data(self, evaluation):

        if self.config.observe_context:
            # exact feature vector
            h = evaluation['h_ex']
        else:
            # feature vector computed previously from sample averages
            h = evaluation['h_ee']

        self.m.add_data(h, evaluation['y'])

