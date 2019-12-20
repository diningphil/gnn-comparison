from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.transforms import ToDense

NUM_SAGE_LAYERS = 3


class SAGEConvolutions(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False,
                 lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)

        if lin is True:
            self.lin = nn.Linear((NUM_SAGE_LAYERS - 1) * hidden_channels + out_channels, out_channels)
        else:
            # GNN's intermediate representation is given by the concatenation of SAGE layers
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, add_loop=False)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, add_loop=False)))
        x3 = self.conv3(x2, adj, mask, add_loop=False)

        x = torch.cat([x1, x2, x3], dim=-1)

        # This is used by GNN_pool
        if self.lin is not None:
            x = self.lin(x)

        return x


class DiffPoolLayer(nn.Module):
    """
    Applies GraphSAGE convolutions and then performs pooling
    """
    def __init__(self, dim_input, dim_hidden, dim_embedding, no_new_clusters):
        """
        :param dim_input:
        :param dim_hidden: embedding size of first 2 SAGE convolutions
        :param dim_embedding: embedding size of 3rd SAGE convolutions (eq. 5, dim of Z)
        :param no_new_clusters: number of clusters after pooling (eq. 6, dim of S)
        """
        super().__init__()
        self.gnn_pool = SAGEConvolutions(dim_input, dim_hidden, no_new_clusters)
        self.gnn_embed = SAGEConvolutions(dim_input, dim_hidden, dim_embedding, lin=False)

    def forward(self, x, adj, mask=None):
        s = self.gnn_pool(x, adj, mask)
        x = self.gnn_embed(x, adj, mask)

        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e


class DiffPool(nn.Module):
    """
    Computes multiple DiffPoolLayers
    """
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        self.max_num_nodes = config.dataset.max_num_nodes
        num_diffpool_layers = config['num_layers']
        gnn_dim_hidden = config['gnn_dim_hidden']  # embedding size of first 2 SAGE convolutions
        dim_embedding = config['dim_embedding']  # embedding size of 3rd SAGE convolutions (eq. 5, dim of Z)
        dim_embedding_MLP = config['dim_embedding_MLP']  # hidden neurons of last 2 MLP layers

        self.num_diffpool_layers = num_diffpool_layers

        # Reproduce paper choice about coarse factor
        coarse_factor = 0.1 if num_diffpool_layers == 1 else 0.25

        gnn_dim_input = dim_features
        no_new_clusters = ceil(coarse_factor * self.max_num_nodes)
        gnn_embed_dim_output = (NUM_SAGE_LAYERS - 1) * gnn_dim_hidden + dim_embedding

        layers = []
        for i in range(num_diffpool_layers):
            diffpool_layer = DiffPoolLayer(gnn_dim_input, gnn_dim_hidden, dim_embedding, no_new_clusters)
            layers.append(diffpool_layer)

            # Update embedding sizes
            gnn_dim_input = gnn_embed_dim_output
            no_new_clusters = ceil(no_new_clusters * coarse_factor)

        self.diffpool_layers = nn.ModuleList(layers)

        # After DiffPool layers, apply again layers of GraphSAGE convolutions
        self.final_embed = SAGEConvolutions(gnn_embed_dim_output, gnn_dim_hidden, dim_embedding, lin=False)
        final_embed_dim_output = gnn_embed_dim_output * (num_diffpool_layers + 1)

        self.lin1 = nn.Linear(final_embed_dim_output, dim_embedding_MLP)
        self.lin2 = nn.Linear(dim_embedding_MLP, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)
        # data = ToDense(data.num_nodes)(data)
        # TODO describe mask shape and how batching works

        # adj, mask, x = data.adj, data.mask, data.x
        x_all, l_total, e_total = [], 0, 0

        for i in range(self.num_diffpool_layers):
            if i != 0:
                mask = None

            x, adj, l, e = self.diffpool_layers[i](x, adj, mask)  # x has shape (batch, MAX_no_nodes, feature_size)
            x_all.append(torch.max(x, dim=1)[0])

            l_total += l
            e_total += e

        x = self.final_embed(x, adj)
        x_all.append(torch.max(x, dim=1)[0])

        x = torch.cat(x_all, dim=1)  # shape (batch, feature_size x diffpool layers)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, l_total, e_total