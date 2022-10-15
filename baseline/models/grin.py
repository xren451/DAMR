import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .BiGRIL import BiGRIL

class Model(nn.Module):
    def __init__(self, d_in=1, d_hidden=64, d_ff=64, d_u=0, d_emb=0, impute_only_holes=True):
        super(Model, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0
        self.d_emb = int(d_emb) if d_emb is not None else 0
        self.impute_only_holes = impute_only_holes

        self.bigrill = BiGRIL(input_size=self.d_in)

    def forward(self, x, mask=None, u=None):
        x = rearrange(x, 'b s n c -> b c n s')

        mask = rearrange(mask, 'b s n c -> b c n s').type(torch.uint8)
        if u is not None:
            u = rearrange(u, 'b s n c -> b c n s')
        imputation = self.bigrill(x, mask=mask, u=u)
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask, x, imputation)
        imputation = torch.transpose(imputation, -3, -1)
        return imputation