import torch
from torch import nn

class SpatialConvOrderK(nn.Module):
    def __init__(self, c_in, c_out, order=2, include_self=True):
        super(SpatialConvOrderK, self).__init__()
        self.include_self = include_self
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.order = order

    def forward(self, x):
        # [batch, features, nodes, steps]
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        return out