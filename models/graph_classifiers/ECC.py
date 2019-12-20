import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree, dense_to_sparse
from torch_geometric.nn import ECConv
from torch_scatter import scatter_add
from utils.batch_utils import _make_block_diag


class ECCLayer(nn.Module):
    def __init__(self, dim_input, dim_embedding, dropout=0.):
        super().__init__()

        fnet1 = nn.Sequential(nn.Linear(1, 16),
                              nn.ReLU(),
                              nn.Linear(16, dim_embedding * dim_input))

        fnet2 = nn.Sequential(nn.Linear(1, 16),
                              nn.ReLU(),
                              nn.Linear(16, dim_embedding * dim_embedding))

        fnet3 = nn.Sequential(nn.Linear(1, 16),
                              nn.ReLU(),
                              nn.Linear(16, dim_embedding * dim_embedding))

        self.conv1 = ECConv(dim_input, dim_embedding, nn=fnet1)
        self.conv2 = ECConv(dim_embedding, dim_embedding, nn=fnet2)
        self.conv3 = ECConv(dim_embedding, dim_embedding, nn=fnet3)

        self.bn1 = nn.BatchNorm1d(dim_embedding)
        self.bn2 = nn.BatchNorm1d(dim_embedding)
        self.bn3 = nn.BatchNorm1d(dim_embedding)

        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(self.bn1(x), p=self.dropout, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(self.bn2(x), p=self.dropout, training=self.training)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.dropout(self.bn3(x), p=self.dropout, training=self.training)

        return x


class ECC(nn.Module):
    """
    Uses fixed architecture.

    IMPORTANT NOTE: we will consider dataset which do not have edge labels.
    Therefore, we avoid learning the function that associates a weight matrix
    to an edge specific weight.

    """

    def __init__(self, dim_features, dim_target, config):
        super().__init__()
        self.config = config
        self.dropout = config['dropout']
        self.dropout_final = config['dropout_final']
        self.num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']

        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            dim_input = dim_features if i == 0 else dim_embedding
            layer = ECCLayer(dim_input, dim_embedding, dropout=self.dropout)
            self.layers.append(layer)

        fnet = nn.Sequential(nn.Linear(1, 16),
                             nn.ReLU(),
                             nn.Linear(16, dim_embedding * dim_embedding))

        self.final_conv = ECConv(dim_embedding, dim_embedding, nn=fnet)
        self.final_conv_bn = nn.BatchNorm1d(dim_embedding)

        self.fc1 = nn.Linear(dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def make_block_diag(self, matrix_list):
        mat_sizes = [m.size(0) for m in matrix_list]
        return _make_block_diag(matrix_list, mat_sizes)

    def get_ecc_conv_parameters(self, data, layer_no):
        v_plus_list, laplacians = data.v_plus, data.laplacians

        # print([v_plus[layer_no] for v_plus in v_plus_list])
        v_plus_batch = torch.cat([v_plus[layer_no] for v_plus in v_plus_list], dim=0)

        laplacian_layer_list = [laplacians[i][layer_no] for i in range(len(laplacians))]
        laplacian_block_diagonal = self.make_block_diag(laplacian_layer_list)
        
        if self.config.dataset.name == 'DD':
            laplacian_block_diagonal[laplacian_block_diagonal<1e-4] = 0

        # First layer
        lap_edge_idx, lap_edge_weights = dense_to_sparse(laplacian_block_diagonal)

        # Convert v_plus_batch to boolean
        return lap_edge_idx, lap_edge_weights, (v_plus_batch == 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, layer in enumerate(self.layers):
            # TODO should lap_edge_index[0] be equal to edge_idx?
            lap_edge_idx, lap_edge_weights, v_plus_batch = self.get_ecc_conv_parameters(data, layer_no=i)
            edge_index = lap_edge_idx if i != 0 else edge_index
            edge_weight = lap_edge_weights if i != 0 else x.new_ones((edge_index.size(1), ))

            edge_index = edge_index.to(self.config.device)
            edge_weight = edge_weight.to(self.config.device)

            # apply convolutional layer
            x = layer(x, edge_index, edge_weight)

            # pooling
            x = x[v_plus_batch]
            batch = batch[v_plus_batch]

        # final_convolution
        lap_edge_idx, lap_edge_weight, v_plus_batch = self.get_ecc_conv_parameters(data, layer_no=self.num_layers)

        lap_edge_idx = lap_edge_idx.to(self.config.device)
        lap_edge_weight = lap_edge_weight.to(self.config.device)

        x = F.relu(self.final_conv(x, lap_edge_idx, lap_edge_weight))
        x = F.dropout(self.final_conv_bn(x), p=self.dropout, training=self.training)

        # TODO: is the following line needed before global pooling?
        # batch = batch[v_plus_batch]

        graph_emb = global_mean_pool(x, batch)

        x = F.relu(self.fc1(graph_emb))
        x = F.dropout(x, p=self.dropout_final, training=self.training)

        # No ReLU specified here todo check with source code (code is not so clear)
        x = self.fc2(x)

        return x
