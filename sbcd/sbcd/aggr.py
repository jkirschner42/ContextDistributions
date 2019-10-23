import numpy as np
from scipy import stats
import os

def aggr(experiment):
    """
    Aggregate data from experiment for plotting.
    So far takes: regret,
    """

    data_list = []

    for item in experiment.parts:
        group = experiment.hdf5[str(item.id)]
        data = aggregate_plot_data_part(item, group)
        save_aggregated_data(experiment, data, item.id)
        data_list.append(data)
        print(f"completed group {item.id}")

    return data_list


def aggregate_plot_data_part(item, group):
    group_data = {}

    group_data['algorithm'] = group.attrs['algorithm']
    group_data['label'] = item.label
    group_data['config'] = item.config
    group_data['group_id'] = item.id
    # group_data['beta'] = item.config['model']['beta']
    group_data['repetitions'] = len(group)

    regret_values = []
    # calculate regret
    for run_id in group:
        dset = group[run_id]
        if len(dset) == 0:
            continue

        T = len(dset)
        if 'T' in group_data:
            if T != group_data['T']:
                print(f"Warning: Potentially incomplete dataset group={item.id}, run={run_id}.")
        else:
            group_data['T'] = T

        regret = 0
        for t, row in enumerate(dset):
            regret += row['y_max'] - row['y_exact']

            if len(regret_values) <= t:
                regret_values.append([])
            regret_values[t].append(regret)

    if len(regret_values) > 0:
        group_data['regret_avg'] = np.mean(regret_values, axis=1)
        group_data['regret_err'] = stats.sem(regret_values, axis=1)
        group_data['regret_std'] = np.std(regret_values, axis=1)

    #     data_tmp = np.load(os.path.join(item.path, f'../../data/aggregated_{item.id}.npy')).item()
    #     group_data['regret_avg'] = data_tmp['regret_avg']
    #     group_data['regret_err'] = data_tmp['regret_sterr']
    #     # # group_data['regret_std'] = np.sqrt(len(group))*data_tmp['regret_sterr']

    return group_data


def save_aggregated_data(experiment, data, id):
    path = experiment.directory
    np.save(os.path.join(path, 'data', f'aggregated_{id}.npy'), data)