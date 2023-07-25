"""
Wrapper function for RNNCell and GRUCell, so that its format is the same as RNN and GRU.
"""
import torch
import torch.nn as nn

class RNNWrapper(nn.Module):
    def __init__(self, rnn):
        super(RNNWrapper, self).__init__()
        self.rnn = rnn
        self._patch()

        # Determine what kind of rnn this is
        if type(self.rnn) in [nn.RNN, nn.GRU, nn.LSTM]:
            self._forward_impl = self.forward_rnn
        elif type(self.rnn) in [nn.RNNCell, nn.GRUCell]:
            self._forward_impl = self.forward_cell
        
        # use GRUCellWithGates if it is GRUCell
        if isinstance(self.rnn, nn.GRUCell):
            new_rnn = GRUCellWithGates(self.rnn.input_size, self.rnn.hidden_size)
            new_rnn.init_weight(self.rnn)
            self.rnn = new_rnn

    def forward(self, x, h): return self._forward_impl(x, h)

    def forward_cell(self, x, h):
        B, T, _ = x.shape
        for t in range(T):
            h = self.rnn(x[:, t, :], h[0])
        return h.reshape(B, 1, -1), h.reshape(1, B, -1)
    
    def forward_rnn(self, x, h):
        return self.rnn(x, h)

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
        hidden_proj = self.weight_hh_layer(hidden_state[:, -10:]) # TODO

        # Split the concatenated gates into separate gates
        r_i, z_i, n_i = torch.split(input_proj, self.hidden_size, dim=1)
        r_h, z_h, n_h = torch.split(hidden_proj, self.hidden_size, dim=1)
        r_bi, z_bi, n_bi = torch.split(self.bias_ih, self.hidden_size, dim=0)
        r_bh, z_bh, n_bh = torch.split(self.bias_hh, self.hidden_size, dim=0)
        r = torch.sigmoid(r_i + r_bi + r_h + r_bh)
        z = torch.sigmoid(z_i + z_bi + z_h + z_bh)
        n = torch.tanh(n_i+ n_bi + r*(n_h + n_bh))

        new_hidden_state = (1 - z) * n + z * hidden_state[:, -10:] # TODO

        self._set_weight_hh()
        self._set_weight_ih()
        return torch.cat([r, z, n, new_hidden_state], dim=1)
    
    def _set_weight_hh(self):
        self.weight_hh = nn.Parameter(self.weight_hh_layer.weight)

    def _set_weight_ih(self):
        self.weight_ih = nn.Parameter(self.weight_ih_layer.weight)

    def init_weight(self, grucell):
        self.weight_hh_layer.weight.data = grucell.weight_hh.data
        self.weight_ih_layer.weight.data = grucell.weight_ih.data
        self.bias_hh.data = grucell.bias_hh.data
        self.bias_ih.data = grucell.bias_ih.data
