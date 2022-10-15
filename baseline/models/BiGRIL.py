import torch
import torch.nn as nn
from .gcrnn import GCGRUCell
from .spatial_conv import SpatialConvOrderK
class BiGRIL(nn.Module):
    def __init__(self,
                 input_size=1,
                 hidden_size=64,
                 ff_size=64,
                 ff_dropout=0.0,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 decoder_order=1,
                 u_size=0,
                 n_nodes=None,
                 embedding_size=0,
                 layer_norm=False,
                 merge='mlp'):
        super(BiGRIL, self).__init__()
        self.fwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            kernel_size=kernel_size,
                            u_size = u_size,
                            n_nodes = None,
                            layer_norm=layer_norm)

        self.bwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            kernel_size=kernel_size,
                            u_size = u_size,
                            n_nodes = None,
                            layer_norm=layer_norm)

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        if merge == 'mlp':
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=input_size,
                          out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1)
            )
        self.supp = None

    def forward(self, x, mask=None, u=None):
        fwd_out, fwd_pred, fwd_repr = self.fwd_rnn(x, mask=mask, u=u)
        imputation = self.out(fwd_out)
        return imputation

class GRIL(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, u_size=None, n_layers=1, kernel_size=2, dropout=0., n_nodes=None, layer_norm=False, decoder_order=1):
        super(GRIL, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)
        self.spatial_decoder = SpatialDecoder(d_in=rnn_input_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              order=decoder_order)
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def forward(self, x, mask=None, u=None, h=None):
        # x:[batch, features, nodes, steps]
        *_, steps = x.size()
        # mask = mask.type(torch.uint8)
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)
        predictions, imputations, states = [], [], []
        representations = []
        #
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]

        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
            # firstly impute missing values with predictions from state
            xs_hat_1 = self.first_stage(h_s)
            # fill missing values in input with prediction
            x_s = torch.where(m_s, x_s, xs_hat_1)
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s)

            x_s = torch.where(m_s, x_s, xs_hat_2)
            inputs = [x_s, m_s]
            inputs = torch.cat(inputs, dim=1)

            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            representations.append(repr_s)

        imputations = torch.stack(imputations, dim=-1)
        predictions = torch.stack(predictions, dim=-1)
        representations = torch.stack(representations, dim=-1)

        return imputations, predictions, representations

    def update_state(self, x, h):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer]))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

class SpatialDecoder(nn.Module):
    def __init__(self, d_in, d_model, d_out, order=1):
        super(SpatialDecoder, self).__init__()
        self.order = order
        self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.graph_conv = SpatialConvOrderK(c_in=d_model, c_out=d_model, order=1, include_self=True)
        self.lin_out = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.read_out = nn.Conv1d(2 * d_model, d_out, kernel_size=1)
        self.activation = nn.PReLU()

    def forward(self, x, m, h, u):
        # [batch, channels, nodes]
        x_in = [x, m, h] if u is None else [x, m, u, h]
        x_in = torch.cat(x_in, 1)
        x_in = self.lin_in(x_in)

        out = self.graph_conv(x_in)
        out = torch.cat([out, h], 1)
        out = self.activation(self.lin_out(out))
        out = torch.cat([out, h], 1)
        return self.read_out(out), out

