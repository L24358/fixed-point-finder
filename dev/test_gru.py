import sys
sys.path.append("../")

import torch
import torch.nn as nn
import numpy as np
import dynamics as dy
from FixedPointFinderTorch import FixedPointFinderTorch
from utils import build_gru_cell

model, args, fps_true = build_gru_cell()
fpf = FixedPointFinderTorch(model)

# grucell = nn.GRUCell(1, 10)
# grucell.weight_hh.data = model.rnn.weight_hh
# grucell.weight_ih.data = model.rnn.weight_ih
# grucell.bias_hh.data = model.rnn.bias_hh
# grucell.bias_ih.data = model.rnn.bias_ih
# h_0 = torch.from_numpy(np.random.uniform(low=-1, high=1, size=40))
# inputs = torch.Tensor([args[0]])

# res1 = h_0[-10:].clone()
# res2 = h_0.clone()
# for _ in range(10):
#     res1 = grucell(inputs.float(), res1.unsqueeze(0).float())[0].squeeze()
#     res2 = model(inputs.reshape(1, 1, -1).float(), res2.reshape(1, 1, -1).float())[0].squeeze()

initial_states = []
for rp in range(20):
    h_0 = np.random.uniform(low=-1, high=1, size=40)
    for _ in range(10): h_0 = dy.gru_sim(h_0, *args) 
    initial_states.append(h_0)
initial_states = np.array(initial_states)
inputs = np.array([args[0]])

unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)
# unique_fps.xstar[0] == fps_true
import pdb; pdb.set_trace()
