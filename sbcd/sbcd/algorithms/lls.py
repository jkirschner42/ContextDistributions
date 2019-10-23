from febo.models import ConfidenceBoundModel
import numpy as np

# define new model
class DoubleRobustLinear(ConfidenceBoundModel):

    def __init__(self, domain):
        super().__init__(domain)
        self.G = np.eye(self.domain.d)  # * self.domain.d * np.log(20/self.config.delta)
        self.G_inv = np.linalg.inv(self.G)
        self.g = np.zeros(self.domain.d)
        self.theta = np.zeros(self.domain.d)
        self.t = 0

    def add_data(self, X, Y, M):
        print(f'added data : {X}, {Y}, {M}')
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        M = np.atleast_2d(M)

        if X.shape[0] == 1:
            if (X == M).all():
                print('skip')
                return

        for x, y, m in zip(X, Y, M):
            b = x - m
            self.G += np.outer(b, b)
            self.g += b * y

        self.t += X.shape[0]  # update data counter
        self.theta = np.linalg.solve(self.G, self.g)
        self.G_inv = np.linalg.inv(self.G)

    def mean(self, X):
        X = np.atleast_2d(X)
        return X.dot(self.theta).reshape(-1, 1)

    def var(self, X):
        X = np.atleast_2d(X)
        # a = np.sum(X * (np.linalg.solve(self.G, X.T).T), axis=1).reshape(-1, 1)
        b = np.sum(X * (self.G_inv.dot(X.T).T), axis=1).reshape(-1, 1)
        # print(a-b)

        return b

    def predictive_var(self, X, Xcond, mcond):
        X = np.atleast_2d(X)
        b = Xcond - mcond
        return np.sum(X * (np.linalg.solve(self.G + np.outer(b, b), X.T).T), axis=1).reshape(-1, 1)

    def mean_var(self, X):
        return self.mean(X), self.var(X)

    def _beta(self):
        # TODO : Use 'exact' value from paper
        return np.sqrt(2)*0.05  # np.sqrt(self.domain.d * np.log(max(self.t,2)/self.config.delta))