import torch
import torch.nn as nn
import numpy as np
import dev.dynamics as dy

def get_rnn_params(rnn): 
    Wi, Wh = rnn.weight_ih, rnn.weight_hh
    bi, bh = rnn.bias_ih, rnn.bias_hh
    return Wh.detach().numpy(), Wi.detach().numpy(), bh.detach().numpy(), bi.detach().numpy()

def get_gru_params(gru): 
    gru.weight_hh.requires_grad = False
    gru.weight_ih.requires_grad = False
    gru.bias_hh.requires_grad = False
    gru.bias_ih.requires_grad = False
    args = [10, *np.split(gru.weight_hh.numpy(), 3), *np.split(gru.weight_ih.numpy(), 3),
            *np.split(gru.bias_hh.numpy(), 3), *np.split(gru.bias_ih.numpy(), 3)]
    return args

def build_rnn_cell():
    model = RNNCellWrapper(1, 10)
    inp = np.random.normal(size=(1,))
    args = get_rnn_params(model.rnn)
    fps = dy.get_fixed_points_by_sim(inp, *args, h=10, name="tanh")
    return model, [inp, *args], fps

def build_gru_cell():
    model = GRUCellWrapper(1, 10)
    inp = np.random.normal(size=(1,))
    args = get_gru_params(model.rnn)
    fps = dy.get_fixed_points_by_sim_gru(inp, model.rnn, dim_h=10)
    return model, [inp, *args], fps

class RNNCellWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCellWrapper, self).__init__()
        self.rnn = nn.RNNCell(input_dim, hidden_dim)
        self._patch()

    def forward(self, x, h):
        B, T, dim_x = x.shape
        for t in range(T):
            h = self.rnn(x[:, t, :], h[0])
        return h.reshape(B, 1, -1), h.reshape(1, B, -1)

    def _patch(self):
        self.batch_first = True
        self.parameters = self.rnn.parameters

class GRUCellWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCellWrapper, self).__init__()
        self.rnn = GRUCellWithGates(input_dim, hidden_dim) #
        self._patch()

    def forward(self, x, h):
        B, T, dim_x = x.shape
        for t in range(T):
            h = self.rnn(x[:, t, :], h[0])
        return h.reshape(B, 1, -1), h.reshape(1, B, -1)

    def _patch(self):
        self.batch_first = True
        self.parameters = self.rnn.parameters

class GRUCellWithGates(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCellWithGates, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear layers for input-to-hidden weights and biases
        self.weight_ih_layer = torch.nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.bias_ih = torch.nn.Parameter(torch.zeros(3 * hidden_size))

        # Linear layers for hidden-to-hidden weights and biases
        self.weight_hh_layer = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.bias_hh = torch.nn.Parameter(torch.zeros(3 * hidden_size))

        self._set_weight_hh()
        self._set_weight_ih()

    def forward(self, input, hidden_state):
        input_proj = self.weight_ih_layer(input)
        hidden_proj = self.weight_hh_layer(hidden_state[:, -self.hidden_size:])

        # Split the concatenated gates into separate gates
        r_i, z_i, n_i = torch.split(input_proj, self.hidden_size, dim=1)
        r_h, z_h, n_h = torch.split(hidden_proj, self.hidden_size, dim=1)
        r_bi, z_bi, n_bi = torch.split(self.bias_ih, self.hidden_size, dim=0)
        r_bh, z_bh, n_bh = torch.split(self.bias_hh, self.hidden_size, dim=0)
        r = torch.sigmoid(r_i + r_bi + r_h + r_bh)
        z = torch.sigmoid(z_i + z_bi + z_h + z_bh)
        n = torch.tanh(n_i+ n_bi + r*(n_h + n_bh))

        new_hidden_state = (1 - z) * n + z * hidden_state[:, -self.hidden_size:]

        self._set_weight_hh()
        self._set_weight_ih()
        return torch.cat([r, z, n, new_hidden_state], dim=1)
    
    def _set_weight_hh(self):
        self.weight_hh = nn.Parameter(self.weight_hh_layer.weight)

    def _set_weight_ih(self):
        self.weight_ih = nn.Parameter(self.weight_ih_layer.weight)


