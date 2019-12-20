import numpy as np


class GraphDataset:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_targets(self):
        targets = [d.y.item() for d in self.data]
        return np.array(targets)

    def get_data(self):
        return self.data

    def augment(self, v_outs=None, e_outs=None, g_outs=None, o_outs=None):
        """
        v_outs must have shape |G|x|V_g| x L x ? x ...
        e_outs must have shape |G|x|E_g| x L x ? x ...
        g_outs must have shape |G| x L x ? x ...
        o_outs has arbitrary shape, it is a handle for saving extra things
        where    L = |prev_outputs_to_consider|.
        The graph order in which these are saved i.e. first axis, should reflect the ones in which
        they are saved in the original dataset.
        :param v_outs:
        :param e_outs:
        :param g_outs:
        :param o_outs:
        :return:
        """
        for index in range(len(self)):
            if v_outs is not None:
                self[index].v_outs = v_outs[index]
            if e_outs is not None:
                self[index].e_outs = e_outs[index]
            if g_outs is not None:
                self[index].g_outs = g_outs[index]
            if o_outs is not None:
                self[index].o_outs = o_outs[index]


class GraphDatasetSubset(GraphDataset):
    """
    Subsets the dataset according to a list of indices.
    """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __getitem__(self, index):
        return self.data[self.indices[index]]

    def __len__(self):
        return len(self.indices)

    def get_targets(self):
        targets = [self.data[i].y.item() for i in self.indices]
        return np.array(targets)
