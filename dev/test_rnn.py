import sys
sys.path.append("../")

import torch
import numpy as np
import dynamics as dy
from FixedPointFinderTorch import FixedPointFinderTorch
from utils import build_rnn_cell

model, args, fps_true = build_rnn_cell()
fpf = FixedPointFinderTorch(model)

initial_states = []
for rp in range(20):
    h_0 = np.random.uniform(low=-1, high=1, size=10)
    for _ in range(10): h_0 = dy.rnn_sim(h_0, *args)
    initial_states.append(h_0)
initial_states = np.array(initial_states)
inputs = np.array([args[0]])

unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)
# unique_fps.xstar[0] == fps_true
import pdb; pdb.set_trace()