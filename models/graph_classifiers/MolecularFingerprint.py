import torch
from torch.nn import ReLU
from torch_geometric.nn import global_add_pool


class MolecularFingerprint(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(MolecularFingerprint, self).__init__()
        hidden_dim = config['hidden_units']

        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_features, hidden_dim), ReLU(),
                                       torch.nn.Linear(hidden_dim, dim_target), ReLU())

    def forward(self, data):
        return self.mlp(global_add_pool(data.x, data.batch))
