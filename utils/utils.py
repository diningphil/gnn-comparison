import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def one_hot(value, num_classes):
    vec = np.zeros(num_classes)
    vec[value - 1] = 1
    return vec


def get_max_num_nodes(dataset_str):
    import datasets
    dataset = getattr(datasets, dataset_str)()

    max_num_nodes = -1
    for d in dataset.dataset:
        max_num_nodes = max(max_num_nodes, d.num_nodes)
    return max_num_nodes
