import torch
import torch.nn as nn

class IdentityGNN(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, node_feat, edges):

        return node_feat
