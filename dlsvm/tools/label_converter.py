import numpy as np


def label_2_pn(data):
    data = data.astype(np.int32)
    n_class = np.unique(data).size
    onehot = np.eye(n_class)[data]

    return np.where(onehot == 1, 1, -1).astype(np.float32)


def label_2_onehot(data):
    data = data.astype(np.int32)
    n_class = np.unique(data).size

    return np.eye(n_class)[data]
