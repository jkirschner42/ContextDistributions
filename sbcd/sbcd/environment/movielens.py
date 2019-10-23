import numpy as np
from febo.environment import ContextMixin
from febo.environment.benchmarks import BenchmarkEnvironment, BenchmarkEnvironmentConfig
from febo.environment.domain import DiscreteDomain
from febo.utils.config import ConfigField, assign_config
import pickle
import os

from sbcd.environment import CDEnvironment, trace_theta


def load_user_data(file):
    users = []
    with open(file) as f:
        for line in f.readlines():
            # remove '\n'
            line = line[:-2]
            # discard postcode and user id
            users.append(line.split('::')[1:-1])

    return users


def group_users_by_gender_and_ages(users):
    d = dict()
    for i, data in enumerate(users):
        gender_age = (data[1], data[2])
        if not gender_age in d:
            d[gender_age] = []
            d[gender_age].append(i)

    return d


def group_users_by_gender_age_occ(users):
    d = dict()
    for i, data in enumerate(users):
        group = (data[0], data[1], data[2])
        if not group in d:
            d[group] = []
        d[group].append(i)

    return d


class Movielens(CDEnvironment):
    def initialize(self, *args, **kwargs):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._user_features = np.load(os.path.join(dir_path, 'movielens/user_features.npy'))
        self._movie_features = np.load(os.path.join(dir_path, 'movielens/movie_features.npy'))
        # self._movie_bias = np.load(os.path.join(dir_path, 'movielens/movie_bias.npy'))
        # self._user_bias = np.load(os.path.join(dir_path, 'movielens/user_bias.npy'))
        self._global_mean = np.load(os.path.join(dir_path, 'movielens/global_mean.npy'))
        self._users = load_user_data(os.path.join(dir_path, 'movielens/users.dat'))

        self._num_users = len(self._users)

        self._users_groups = group_users_by_gender_age_occ(self._users)

        self.d = self._movie_features.shape[1]**2


        self.k = self._movie_features.shape[0]

        self._domain = DiscreteDomain(points=[*range(self.k)], d=self.d)


        # precompute all features
        self._features_cache = dict()
        # for u in range(self._num_users):
        #     for i in range(self.k):
        #         self._features_cache[(u, i)] = np.concatenate((np.outer(self._user_features[u], self._movie_features[i]).flatten(),
        #                                                        [self._user_bias[u], self._movie_bias[i], self._global_mean]))

        # self._theta = np.concatenate((trace_theta(self._movie_features.shape[1]), [1,1,1]))
        self._theta = trace_theta(self._movie_features.shape[1])
        self._x0 = 0  # default action

        # for each group, compute best movie:
        self._best_movie_for_group = {}


        # precompute best action for each group
        # print("computing best movie for each group")
        # for j, (group, users) in enumerate(self._users_groups.items()):
        #     print(j)
        #     best_movie = None
        #     best_rating = -100
        #     for movie in range(self.k):
        #         ratings = []
        #         for user in users:
        #             self._current_user = user
        #             ratings.append(self.f(movie))
        #
        #
        #         avg_rating = np.average(ratings)
        #         if avg_rating > best_rating:
        #             best_movie = movie
        #             best_rating = avg_rating
        #
        #     self._best_movie_for_group[group] = best_movie
        #     print(f"best rating: {best_rating}")
        #
        # with open(os.path.join(dir_path, 'movielens/best_group_occ.pickle'), 'wb') as file:
        #     pickle.dump(self._best_movie_for_group, file, protocol=pickle.HIGHEST_PROTOCOL)
        # exit()

        # load best action for each group
        print("loading best movie for each group")
        with open(os.path.join(dir_path, 'movielens/best_group_occ.pickle'), 'rb') as file:
            self._best_movie_for_group = pickle.load(file)
        print("done")

        return super().initialize(*args, **kwargs)

    def _get_features(self, u, i):
        return np.outer(self._user_features[u], self._movie_features[i]).flatten()

    def get_context(self):
        # choose random user
        self._current_user = np.random.randint(self._num_users)
        self._current_group = tuple(self._users[self._current_user])

        # sample access to context distribution
        def sampler(n):
            if self.config.exact_context:
                return np.array([self._get_features(self._current_user, i) for i in range(self.k)])
            elif n == -1:
                expected_user_features = np.average([self._user_features[u] for u in self._users_groups[self._current_group]], axis=0)
                return np.array([np.outer(expected_user_features, self._movie_features[i]).flatten() for i in range(self.k)])
            else:
                # sample users from the same group
                u_sample = np.random.choice(self._users_groups[self._current_group], size=n)
                return np.array([np.average([self._get_features(u, i) for u in u_sample], axis=0) for i in range(self.k)])

        self._sampler = sampler

        return {'sampler': sampler}

    def f(self, x):
        # compute rating
        # first compute a score and normalize

        score = np.array(self._get_features(self._current_user, x).dot(self._theta))

        # clip score to 0 and 5 and round:
        score = np.min([5, np.max([0, score])])
        score = np.round(2*score)/2
        return score


    def _noise_function(self, x):

        noise = np.random.choice([-1., 0.5, 0., 0.5, 1.])

        # clip noise
        if self.f(x) + noise > 5:
            return 0.

        if self.f(x) + noise < 0.:
            return 0.

        return noise


    def _init_noise_function(self):
        pass

    def evaluate(self, x=None):
        evaluation = super().evaluate(x)
        # provide exact feature
        evaluation['h_ex'] = self._get_features(self._current_user, x)
        return evaluation

    @property
    def max_value(self):
        # return 5.
        return self.f(self._best_movie_for_group[self._current_group])

    def _get_dtype_fields(self):
        dtype_fields = []

        # replace dtype for x
        for t in super()._get_dtype_fields():
            if t[0] == 'x':
                dtype_fields.append(('x', 'i8'))
            else:
                dtype_fields.append(t)

        return dtype_fields + [('h_ex', f'({self.d},)f8')]
