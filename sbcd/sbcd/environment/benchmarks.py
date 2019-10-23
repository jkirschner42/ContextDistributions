import numpy as np
from febo.environment.domain import DiscreteDomain

from sbcd.environment import CDEnvironment



class CDBandit2(CDEnvironment):
    def initialize(self, *args, **kwargs):
        """
        The function is f(x,c) = sum_i (x_i - c_i)^2
        """
        self.d1 = 5 # feature dimension for actions
        # self.d2 = 4  # feature dimension for context

        self.d = 3*self.d1
        self.k = self.config.num_domain_points  # number of actions

        self._domain = DiscreteDomain(points=[*range(self.k)], d=self.d)

        # parameter
        self._theta = np.ones(self.d)
        self._theta[-self.d1:] = -2.

        # create action features
        self.actions = np.abs(np.random.normal(size=self.d1 * self.k)).reshape(self.k, self.d1)

        self._x0 = 0  # default action
        return super().initialize(*args, **kwargs)

    def _get_features(self, x, c):
        return np.concatenate((x*x, c*c, x*c))

    def get_context(self):
        self._current_mean = np.random.normal(size=self.d1)

        # sample context realization
        self.current_context = self._current_mean + np.random.normal(size=self.d1)

        # sample access to context distribution
        def sampler(n):
            if self.config.exact_context:
                return np.array([self._get_features(x, self.current_context) for x in self.actions])
            elif n == -1:
                # compute expected context
                return np.array([
                    np.concatenate((
                        x*x,
                        self._current_mean*self._current_mean + 1,  # EE[c*c] = Var(c) + 1
                        x*self._current_mean
                    ))
                    for x in self.actions
                ])
            else:
                sample = self._current_mean + np.random.normal(size=self.d1*n).reshape(n,self.d1)
                return np.array([np.average([self._get_features(x, c) for c in sample], axis=0) for x in self.actions])

        self._sampler = sampler
        return {'sampler': sampler}

    def f(self, x):
        # Note: the parameter x is the action index here
        # sample context realization
        return self._get_features(self.actions[x], self.current_context).dot(self._theta)

    def evaluate(self, x=None):
        # Note: the parameter x is the action index here
        evaluation = super().evaluate(x)
        # provide exact feature
        evaluation['h_ex'] = self._get_features(self.actions[x], self.current_context)
        return evaluation

    @property
    def max_value(self):
        # in this case, argmax E[f(i)] = argmax f(i)
        # compute best action
        res = np.sum(self.actions**2 - 2*self.actions*self._current_mean, axis=1)
        return self.f(np.argmax(res))

    def _get_dtype_fields(self):
        dtype_fields = []

        # replace dtype for x
        for t in super()._get_dtype_fields():
            if t[0] == 'x':
                dtype_fields.append(('x', 'i8'))
            else:
                dtype_fields.append(t)

        return dtype_fields + [('h_ex', f'({self.d},)f8')]



class CDBandit1(CDEnvironment):
    def initialize(self, *args, **kwargs):
        self.d1 = 4  # feature dimension for actions
        self.d2 = 4  # feature dimension for context

        self.d = self.d1 + self.d2  # total number of features
        self.k = self.config.num_domain_points  # number of actions

        self._domain = DiscreteDomain(points=[*range(self.k)], d=self.d)

        # parameter
        self._theta = np.ones(self.d)/np.sqrt(self.d)

        # create action features
        self.action_features = np.abs(np.random.normal(size=self.d1*self.k)).reshape(self.k, self.d1)

        self._x0 = 0  # default action
        return super().initialize(*args, **kwargs)

    def get_context(self):
        self._current_mean = np.random.uniform(size=self.d2)


        # # sample context realization
        if self.config.exact_context:
            self.current_context = self._current_mean
        else:
            self.current_context = np.random.multivariate_normal(self._current_mean, np.eye(self.d2))

        # sample access to context distribution
        def sampler(n):
            Phi = np.empty((self.k, self.d))
            if self.config.exact_context:
                h2 = np.repeat(self._current_mean.reshape(-1, 1), n, axis=1).T
            else:
                h2 = np.random.multivariate_normal(self._current_mean, np.eye(self.d2), size=n)

            for i in range(self.k):
                # returns `n` samples for action feature `i` under the current context distribution
                h1 = self.action_features[i]
                Phi[i] = np.concatenate((h1, np.average(h2, axis=0)))

            return Phi

        self._sampler = sampler

        return {'sampler': sampler}

    def f(self, x):
        # sample context realization
        h = np.concatenate((self.action_features[x], self.current_context))
        return h.dot(self._theta)

    def evaluate(self, x=None):
        evaluation = super().evaluate(x)
        # provide exact feature
        evaluation['h_ex'] = np.concatenate((self.action_features[x], self.current_context))
        return evaluation

    @property
    def max_value(self):
        # in this case, argmax E[f(i)] = argmax f(i)
        return np.max([self.f(i) for i in range(self.k)])

    def _get_dtype_fields(self):
        dtype_fields = []

        # replace dtype for x
        for t in super()._get_dtype_fields():
            if t[0] == 'x':
                dtype_fields.append(('x', 'i8'))
            else:
                dtype_fields.append(t)

        return dtype_fields + [('h_ex', f'({self.d},)f8')]
