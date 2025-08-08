import torch
from torch.nn import Parameter
import math

from .graph import GCNGraph
from .mgcnfunc import MGCNFunction

class GraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.weights = Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        # if self.weights is not None:
        #     nn.init.xavier_uniform_(self.weights)
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, g: GCNGraph, x: torch.Tensor) -> torch.Tensor:
        return MGCNFunction.apply(g, x, self.weights)
    
    