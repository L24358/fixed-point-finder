import sys
sys.path.append("../")

import torch
import torch.nn as nn
import numpy as np
from FixedPointFinderTorch import FixedPointFinderTorch
from dev.wrapper import RNNWrapper

import dev.dynamics as dy
from dev.utils import get_rnn_params, get_gru_params

########## Test RNN Cell ##########
if False:
    # Build cell, Wrapper(cell), FixedPointFinder
    h = 10
    rnncell = nn.RNNCell(1, h)
    model = RNNWrapper(rnncell)
    fpf = FixedPointFinderTorch(model)

    # get ground truth fp
    inp = np.random.normal(size=(1,))
    args = get_rnn_params(model.rnn)
    fps_true = dy.get_fixed_points_by_sim(inp, *args, dim_h=h)

    # get initial states and inputs
    initial_states = []
    for rp in range(20):
        h_0 = np.random.uniform(low=-1, high=1, size=h)
        for _ in range(10): h_0 = dy.rnn_sim(h_0, inp, *args) 
        initial_states.append(h_0)
    initial_states = np.array(initial_states)
    inputs = np.array([inp])

    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)
    print("Are the fixed points for RNNCell correct? ", not dy.different(unique_fps.xstar[0], fps_true)[0])

########## Test GRU Cell ##########
if True:
    # Build cell, Wrapper(cell), FixedPointFinder
    h = 10
    # rnncell = nn.GRUCell(1, h)
    # model = RNNWrapper(rnncell)
    from utils import GRUCellWrapper
    model = GRUCellWrapper(1, 10)
    fpf = FixedPointFinderTorch(model)

    # get ground truth fp
    inp = np.random.normal(size=(1,))
    args = get_gru_params(model.rnn)
    fps_true = dy.get_fixed_points_by_sim_gru(inp, model.rnn, dim_h=h)

    # get initial states and inputs
    initial_states = []
    for rp in range(20):
        h_0 = np.random.uniform(low=-1, high=1, size=4*h)
        for _ in range(10): h_0 = dy.gru_sim(h_0, inp, *args) 
        initial_states.append(h_0)
    initial_states = np.array(initial_states)
    inputs = np.array([inp])

    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)
    # print("Are the fixed points for GRUCell correct? ", not dy.different(unique_fps.xstar[0], fps_true)[0])

import pdb; pdb.set_trace()