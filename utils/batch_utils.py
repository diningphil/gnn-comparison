import torch
import numpy as np
import networkx as nx

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, scatter_, to_dense_adj


def construct_mask_indices(sizes):
    num_rows, num_cols = sum(sizes), len(sizes)

    indices = []
    for i, size in enumerate(sizes):
        cum_sum = sum(sizes[:i])
        indices.append((cum_sum, cum_sum + size))
    return indices


def _make_block_diag(mats, mat_sizes):
    block_diag = torch.zeros(sum(mat_sizes), sum(mat_sizes))

    for i, (mat, size) in enumerate(zip(mats, mat_sizes)):
        cum_size = sum(mat_sizes[:i])
        block_diag[cum_size:cum_size+size,cum_size:cum_size+size] = mat

    return block_diag


def make_block_diag(data):
    data = data.to_data_list()
    adjs = [to_dense_adj(d.edge_index).squeeze(0) for d in data]
    adj_sizes = [a.size(0) for a in adjs]
    bd_mat = _make_block_diag(adjs, adj_sizes)
    mask_indices = construct_mask_indices(adj_sizes)
    return bd_mat, mask_indices


def get_adj(block_diag, index):
    from_i, to_i = index
    return block_diag[from_i:to_i, from_i:to_i]


def mock_batch(batch_size):
    """construct pyG batch"""
    graphs = []
    while len(graphs) < batch_size:
        G = nx.erdos_renyi_graph(np.random.choice([300, 500]), 0.5)
        if G.number_of_edges() > 1:
            graphs.append(G)

    adjs = [torch.from_numpy(nx.to_numpy_array(G)) for G in graphs]
    graph_data = [dense_to_sparse(A) for A in adjs]
    data_list = [Data(x=x, edge_index=e) for (e, x) in graph_data]
    return Batch.from_data_list(data_list)


def test():
    batch_size = 3
    data = mock_batch(batch_size=batch_size)

    # create block diagonal matrix of batch
    # block size: [nodes_in_batch] x [nodes_in_batch]
    block_diag, indices = make_block_diag(data)
    for i in range(batch_size):
        graph_adj = get_adj(block_diag, indices[i])
        print(graph_adj)