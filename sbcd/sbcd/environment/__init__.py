# from .agroscope import AgroscopeCrops
from febo.environment import ContextMixin
from febo.environment.benchmarks import BenchmarkEnvironment, BenchmarkEnvironmentConfig
from febo.utils.config import ConfigField, assign_config
import numpy as np

class CDBanditConfig(BenchmarkEnvironmentConfig):
    exact_context = ConfigField(False)

@assign_config(CDBanditConfig)
class CDEnvironment(ContextMixin, BenchmarkEnvironment):
    pass


def trace_theta(d):
    """
    compute a theta such that for a dxd Matrix A
    trace(A) = A.flatten().dot(theta)
    :param d:  dimension
    :return:  theta
    """
    theta = np.zeros(d*d)
    cur = 0
    for i in range(d):
        theta[cur] = 1
        cur += d+1
    return theta
