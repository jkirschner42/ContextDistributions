from febo.plots import Plot
from febo.utils import get_logger
from febo.utils.config import config_manager
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

logger = get_logger('plots')

class Beta(Plot):

    def plot(self, show=True, group_id=None, run_id=None):
        hdf5 = self.experiment.hdf5
        performance = {}

        T = config_manager.get_setting('controller', 'T')
        logger.info(f'Reading values at time T={T}')
        # if isinstance(self.experiment, MultiExperiment):

        for item in self.experiment.parts:
            dset_group = hdf5[str(item.id)]
            algorithm = dset_group.attrs['algorithm']
            # if 'algorithm.bose' in item.config:
            #     algorithm += f"-{item.config['algorithm.bose'].get('compute_method', '')}"
            # if 'algorithm.ids' in item.config:
            #     algorithm += f"-{item.config['algorithm.ids'].get('compute_method', '')}"
            #     algorithm += f"-{item.config['algorithm.ids'].get('use_1pa', '')}"

            # initialize
            if not algorithm in performance:
                performance[algorithm] = ([], [], [])

            # get beta from config
            beta = item.config['model']['beta']

            # read all values at time T
            values = []
            for dset in dset_group:
                dset = dset_group[dset]
                regret = np.sum(dset[:T]['y_max'] - dset[:T]['y_exact'])
                values.append(regret)

            # record performance
            performance[algorithm][0].append(beta)
            performance[algorithm][1].append(np.average(values))
            performance[algorithm][2].append(scipy.stats.sem(values))


        for algorithm, values in performance.items():
            x_data = values[0]
            y_data = values[1]

            # jointly sort
            x_data, y_data = zip(*sorted(zip(x_data, y_data)))
            plt.errorbar(x_data, y_data, yerr=values[2], label=algorithm, marker='*')

        plt.legend()
        path = os.path.join(self.experiment.directory, 'plots')
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, 'beta_plot.png')
        plt.savefig(path)

        if show:
            plt.show()