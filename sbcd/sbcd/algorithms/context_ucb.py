from febo.algorithms import Algorithm
from febo.models import LinearLeastSquares
import numpy as np

class ContextualUCB(Algorithm):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.m = LinearLeastSquares(self.domain)  # initialize model

    def _next(self, context):
        points = context['domain'].points
        ucb = self.m.ucb(points)
        max_ucb = np.max(ucb)
        return points[(ucb == max_ucb).flatten()][0]

    def add_data(self, evaluation):
        self.m.add_data(evaluation['x'], evaluation['y'])