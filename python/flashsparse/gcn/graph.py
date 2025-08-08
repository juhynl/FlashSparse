import torch
from torch_geometric.utils import add_self_loops, get_laplacian
import warnings
import numpy as np
from scipy import sparse as sp

import FS_Block

class GCNGraph:
    def __init__(
        self,
        row_pointers: torch.Tensor,
        column_index: torch.Tensor,
        degrees: torch.Tensor,
        t_window_rowTensor: torch.Tensor,
        t_atomicTensor: torch.Tensor
    ):
        self.row_pointers = row_pointers
        self.column_index = column_index
        self.degrees = degrees
        self.t_window_rowTensor = t_window_rowTensor
        self.t_atomicTensor = t_atomicTensor
    
    def to(self, device):
        self.row_pointers = self.row_pointers.to(device)
        self.column_index = self.column_index.to(device)
        self.degrees = self.degrees.to(device)
        self.t_window_rowTensor = self.t_window_rowTensor.to(device)
        self.t_atomicTensor = self.t_atomicTensor.to(device)
        return self
    
    def __str__(self):
        return f"GCNGraph(row_pointers={self.row_pointers.shape}, column_index={self.column_index.shape}, degrees={self.degrees.shape}, t_window_rowTensor={self.t_window_rowTensor.shape}, t_atomicTensor={self.t_atomicTensor.shape})"

def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0

def graph(edge_index: torch.Tensor, num_nodes: int):
    num_nodes=num_nodes+16-(num_nodes%16)
    
    edge_index = edge_index.to('cpu')
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    column_index = torch.IntTensor(adj.indices)
    row_pointers = torch.IntTensor(adj.indptr)
    dd = (row_pointers[1:] - row_pointers[:-1]).tolist()
    dd=torch.tensor(dd, dtype=torch.float32) 
    dd= torch.rsqrt(dd) 
    
    dd = (row_pointers[1:] - row_pointers[:-1]).tolist()
    dd=torch.tensor(dd, dtype=torch.float32) 
    dd= torch.rsqrt(dd) 
    
    row_pointers, \
    column_index, \
    degrees, \
    t_window_rowTensor, \
    t_atomicTensor = FS_Block.blockProcess_tf32_balance(row_pointers, column_index, dd, 8, 4, 32)

    g= GCNGraph(
        row_pointers=row_pointers,
        column_index=column_index,
        degrees=degrees,
        t_window_rowTensor=t_window_rowTensor,
        t_atomicTensor=t_atomicTensor
    )

    return g