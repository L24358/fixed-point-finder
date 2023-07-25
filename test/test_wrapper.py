import sys
sys.path.append("../")

import torch
import torch.nn as nn
import numpy as np
from FixedPointFinderTorch import FixedPointFinderTorch

# Build cell, Wrapper(cell), FixedPointFinder
h = 10
model = nn.RNNCell(1, h)
fpf = FixedPointFinderTorch(model)

# get initial states, input
inputs = torch.randn(1, 1)
initial_states = []
for _ in range(20):
    h_0 = torch.rand(1, h)
    for _ in range(10): h_0 = model(inputs, h_0) 
    initial_states.append(h_0.detach().squeeze().numpy())

inputs = inputs.numpy()
initial_states = np.array(initial_states)
unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

import pdb; pdb.set_trace()