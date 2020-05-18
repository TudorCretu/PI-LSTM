import numpy as np

import h5py


def load_data(case, noise):
    if case == 'MFE':
        with h5py.File("data/MFE.h5", 'r') as hf:
            time = np.array(hf.get('/t'))
            data = np.array(hf.get('/u'))
    elif case == 'Lorenz':
        with h5py.File("data/Lorenz_rho_28.0_sigma_10.0_beta_2.67_noise_{}.h5".format(noise), 'r') as hf:
            time = np.array(hf.get('t'))
            x = np.array(hf.get('x'))
            y = np.array(hf.get('y'))
            z = np.array(hf.get('z'))
            data = np.array((x, y, z)).T
    else:
        raise ValueError('Invalid data system case')

    return time, data



def split_data(dataset, target, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        data.append(dataset[i-history_size:i])
        labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)
