import numpy as np
from .acquisition import AcquisitionAlgorithm
from .model import ModelMixin
from scipy.optimize import  minimize, brentq
import scipy

class UCB(ModelMixin, AcquisitionAlgorithm):
    """
    Implements the Upper Confidence Bound (UCB) algorithm.
    """

    def initialize(self, **kwargs):
        super(UCB, self).initialize(**kwargs)

    def acquisition(self, X):
        X = X.reshape(-1, self.domain.d)
        return -(self.model.ucb(X))

    def acquisition_grad(self, x):
        mean, var = self.model.mean_var(x)
        std = np.sqrt(var)
        dmu_dX, dv_dX = self.model.mean_var_grad(x)
        dmu_dX = dmu_dX.reshape(dmu_dX.shape[0:2]) # flatten out inner dimension
        return -(mean + self.model.beta * std), -(dmu_dX + self.model.beta  * dv_dX) / (2 * std)

