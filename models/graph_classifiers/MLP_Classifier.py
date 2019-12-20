import torch
import torch.nn.functional as F
from torch.nn import Linear


class MLPClassifier(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(MLPClassifier, self).__init__()

        hidden_units = config['hidden_units']

        self.fc_global = Linear(dim_features, hidden_units)
        self.out = Linear(hidden_units, dim_target)

    def forward(self, x, batch):
        return self.out(F.relu(self.fc_global(x)))
