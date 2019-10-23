import numpy as np
from febo.environment import DiscreteDomain
from febo.utils import parse_int_set, get_logger
from sbcd.environment import CDEnvironment, trace_theta

import os
import pickle

logger = get_logger('environment')


class AgroscopeCrops(CDEnvironment):

    def initialize(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # load crop data
        data_dict = np.load(os.path.join(dir_path, 'agroscope/agroscope_all.npy'), allow_pickle=True).item()
        self.yield_data = data_dict['yield_data']
        self.context_data = data_dict['context_data']
        self.site_year = data_dict['site_year']
        self.varieties = data_dict['varieties']
        self.sites = data_dict['sites']

        self.k = len(self.varieties)
        self._site_features = np.eye(len(self.sites))

        # load features
        self.V = np.load(os.path.join(dir_path, 'agroscope/variety_features.npy'), allow_pickle=True)
        self.W = np.load(os.path.join(dir_path, 'agroscope/site_weather_features.npy'), allow_pickle=True)
        self.noise = np.load(os.path.join(dir_path, 'agroscope/noise.npy'), allow_pickle=True)
        self.y_mean, self.y_std = np.load(os.path.join(dir_path, 'agroscope/normalization.npy'), allow_pickle=True)

        # normalize yield data:
        self.yield_data[:,3] -= self.y_mean
        self.yield_data[:,3] /= self.y_std

        # feature dimension
        self.d = self.V.shape[1] ** 2

        # trace parameter
        self._theta = trace_theta(self.V.shape[1])

        self._domain = DiscreteDomain(points=[*range(self.k)], d=self.d)

        # compute distribution over suitability factor per site
        num_weather_features = len(self.context_data[0][2:])
        self._weather_var = np.empty(shape=(len(self.sites), num_weather_features))
        self._weather_mean = np.empty(shape=(len(self.sites), num_weather_features))

        for i, site in enumerate(self.sites):
            features = self.context_data[self.context_data[:,0] == site][:,2:]
            self._weather_var[i] = np.var(features, axis=0)
            self._weather_mean[i] = np.mean(features, axis=0)


        # precompute best crop for each site,year
        # self._best_crop_for_site = dict()
        # print("computing best crop for each site")
        # for site in self.sites:
        #     best_crop = None
        #     best_yield = -10e10
        #     for crop in self.varieties:
        #         # get all yields for crop on site
        #         yields = self.yield_data[np.all(self.yield_data[:,[0,2]] == (site, crop), axis=1)][:, 3]
        #
        #         if len(yields) == 0:
        #             continue  # crop was never planted on that site
        #
        #         avg_yield = np.mean(yields)
        #         if avg_yield > best_yield:
        #             best_yield = avg_yield
        #             best_crop = self.varieties.index(crop)
        #
        #     self._best_crop_for_site[site] = best_crop
        # with open(os.path.join(dir_path, 'agroscope/best_crop_for_site.pickle'), 'wb') as file:
        #     pickle.dump(self._best_crop_for_site, file, protocol=pickle.HIGHEST_PROTOCOL)
        # print("done")
        # exit()

        # print("loading best crop for each site")
        # with open(os.path.join(dir_path, 'agroscope/best_crop_for_site.pickle'), 'rb') as file:
        #     self._best_crop_for_site = pickle.load(file)
        # print("done")

        # self._best_crop_for_site_year = dict()
        # print("computing best crop for each site and year")
        # for site, year in self.site_year:
        #     best_crop = None
        #     best_yield = -10e10
        #     for crop in self.varieties:
        #         # get all yields for crop on site
        #         yields = self.yield_data[np.all(self.yield_data[:, :3] == (site, year, crop), axis=1)][:, 3]
        #
        #         if len(yields) == 0:
        #             continue  # crop was never planted on that site
        #
        #         avg_yield = np.mean(yields)
        #         if avg_yield > best_yield:
        #             best_yield = avg_yield
        #             best_crop = self.varieties.index(crop)
        #
        #     self._best_crop_for_site_year[(site,year)] = best_crop
        #
        # with open(os.path.join(dir_path, 'agroscope/best_crop_for_site_year.pickle'), 'wb') as file:
        #     pickle.dump(self._best_crop_for_site_year, file, protocol=pickle.HIGHEST_PROTOCOL)
        # print("done")
        # exit()
        #

        # print("loading best crop for each site and year")
        # with open(os.path.join(dir_path, 'agroscope/best_crop_for_site_year.pickle'), 'rb') as file:
        #     self._best_crop_for_site_year = pickle.load(file)
        # print("done")

        # self._best_crop_for_site_year = dict()
        # print("computing best crop for each site and year")
        # for site, year in self.site_year:
        #     best_crop = None
        #     best_yield = -10e10
        #     for crop in range(self.k):
        #         # get all yields for crop on site
        #         crop_yield = self._theta.dot(self._get_features(crop, site, year))
        #         if crop_yield > best_yield:
        #             best_yield = crop_yield
        #             best_crop = crop
        #
        #     self._best_crop_for_site_year[(site, year)] = best_crop
        #
        # with open(os.path.join(dir_path, 'agroscope/best_crop_for_site_year_model.pickle'), 'wb') as file:
        #     pickle.dump(self._best_crop_for_site_year, file, protocol=pickle.HIGHEST_PROTOCOL)
        # print("done")
        # exit()

        print("loading best crop for each site and year")
        with open(os.path.join(dir_path, 'agroscope/best_crop_for_site_year_model.pickle'), 'rb') as file:
            self._best_crop_for_site_year = pickle.load(file)
        print("done")

        self._x0 = 0
        return super().initialize()

    def _get_data(self, year, site, var=None):
        data = self._data[np.logical_and(self._data.year == year, self._data.siteId == site)]
        if not var is None:
            data = data[data.finalListNumber == var]
        return data

    def _get_site_weather_features(self, site, year):
        # augment suitability factors with one-hot encoding per site
        weather_features = self.context_data[np.all(self.context_data[:, :2] == (site, year), axis=1)][0, 2:].astype('float')

        return np.concatenate((
            self._site_features[self.sites.index(site)],
            weather_features
        ))

    def _get_features(self, variety, site, year):
        return np.outer(self.V[variety], self._get_site_weather_features(site, year).dot(self.W)).flatten().astype('float')

    def _get_features_from_suitability_factors(self, variety, site, factors):
        return np.outer(self.V[variety], np.concatenate((self._site_features[site], factors)).dot(self.W)).flatten().astype('float')

    def get_context(self):
        # randomly pick site_year combination
        self.current_site_year = self.site_year[np.random.randint(len(self.site_year))]
        self.current_site = self.current_site_year[0]
        site_index = self.sites.index(self.current_site)
        # one hot encoding for site
        self.current_site_features = self._get_site_weather_features(*self.current_site_year)
        self._current_suitability_features = self.current_site_features[len(self.sites):]


        # sample a "prediction"
        predicted_suitability_features = np.random.multivariate_normal(self._current_suitability_features,
                                              np.diag(self._weather_var[site_index]))



        def sampler(n):
            # # zero-out features if no data is available
            # if len(self.get_current_yields(i)) == 0:
            #     return np.zeros((n, self.d))

            if self.config.exact_context:
                return np.array([self._get_features(i, *self.current_site_year) for i in range(self.k)])
            elif n == -1:
                # expected context:
                return np.array(
                    [self._get_features_from_suitability_factors(i, self.sites.index(self.current_site), predicted_suitability_features) for i in range(self.k)]
                )
            else:
                # Sample "predictions" for the site's suitability factors

                w_sample = np.random.multivariate_normal(predicted_suitability_features,
                                              np.diag(self._weather_var[site_index]),
                                              size=n)

                # compute embeddings for each sampled suitability factor
                return np.array([np.average([self._get_features_from_suitability_factors(i, site_index, w) for w in w_sample], axis=0) for i in range(self.k)])

        return {'sampler': sampler}

    def f(self, x):
        # get yield measurements for x in current_site_year
        return np.array(self._theta.dot(self._get_features(x, *self.current_site_year)))


    def evaluate(self, x=None):
        evaluation = super().evaluate(x)
        # provide exact feature
        evaluation['h_ex'] = self._get_features(x, *self.current_site_year)
        return evaluation


    def _init_noise_function(self):
        pass

    def _noise_function(self, x):
        return np.random.choice(self.noise)
        # yields = self.get_current_yields(x)
        # if len(yields) == 0:
        #     return 0
        #
        # average = np.mean(yields)
        # return np.random.choice(yields) - average

    def get_current_yields(self, x):
        return self.yield_data[np.all(self.yield_data[:,:3] == (*self.current_site_year, self.varieties[x]),axis=1)][:,3]

    @property
    def max_value(self):
        return self.f(self._best_crop_for_site_year[(self.current_site_year[0], self.current_site_year[1])])


    def _get_dtype_fields(self):
        dtype_fields = []

        # replace dtype for x
        for t in super()._get_dtype_fields():
            if t[0] == 'x':
                dtype_fields.append(('x', 'i8'))
            else:
                dtype_fields.append(t)

        return dtype_fields + [('h_ex', f'({self.d},)f8')]
