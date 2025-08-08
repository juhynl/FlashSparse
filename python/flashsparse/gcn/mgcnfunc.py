#!/usr/bin/env python3
import torch
import sys
import math
import time 
import torch.nn as nn
from torch.nn.parameter import Parameter

import FS_SpMM


n_heads = 8
n_output = 8

def gen_test_tensor(X_prime):
    n_rows = X_prime.size(0)
    n_cols = X_prime.size(1)
    
    X_new = []
    for i in range(n_rows):
        tmp = [i] * n_cols
        X_new.append(tmp)

    X_new = torch.FloatTensor(X_new).cuda()
    return X_new



class MGCNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, x, weights):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(x, weights)
        ctx.graph = g

        # GEMM node update
        X_prime = torch.mm(x, weights)

        # SpMM
        X_prime  = FS_SpMM.forward_tf32_gnn_acc(   
            g.row_pointers, 
            g.column_index, 
            g.degrees, 
            g.t_window_rowTensor,
            g.t_atomicTensor,
            X_prime, 
            (g.row_pointers.size(0) - 1), 
            X_prime.size(1), 
            X_prime.size(0))[0]
        # print("==========After Aggreation=========")
        # print(X_prime)
        # sys.exit(0)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        x, weights = ctx.saved_tensors
        g = ctx.graph
        # SPMM backward propagation.
        d_input_prime  = FS_SpMM.forward_tf32_gnn_acc(   
                    g.row_pointers, 
                    g.column_index, 
                    g.degrees, 
                    g.t_window_rowTensor,
                    g.t_atomicTensor,
                    d_output, 
                    (g.row_pointers.size(0) - 1), 
                    d_output.size(1), 
                    d_output.size(0))[0]
        # GEMM backward propagation.
        d_input = torch.mm(d_input_prime, weights.t())
        d_weights = torch.mm(x.t(), d_input_prime)
        return None, d_input, d_weights

