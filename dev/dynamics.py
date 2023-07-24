import torch
import numpy as np
import torch.nn as nn
from scipy.optimize import fsolve

########################################################
#                  Elementary Functions                #
########################################################

def different(x1, x2, threshold=1e-7):
    """See if ``x1``, ``x2`` are sufficiently different."""
    diff = np.array(x1) - np.array(x2)
    norm = pow(diff, 2).sum()
    if norm > threshold: return True, norm
    else: return False, norm

def relu(x):
    return np.maximum(x, 0)

def drelu(x):
    """Derivative of relu function."""
    return np.where(x > 0, 1, 0)

def dclip(x, maxx=5):
    """Derivative of clip function."""
    return np.where(np.logical_and(x > 0, x < 5), 1, 0)

def get_nonlinearity(name):
    if name == "tanh": return np.tanh
    elif name == "relu": return relu
    elif name == "clip(tanh)": return lambda x: np.clip(np.tanh(x), 0, 5)
    else: raise NotImplementedError(f"nonlinearity of type {name} is not implemented.")

########################################################
#                          RNN                         #
########################################################

def rhs(h, *args, **kwargs):
    '''rhs - lhs of the RNN, i.e. tanh(...) - h = 0.'''
    kw = {"name": "tanh"}
    kw.update(kwargs)

    x, rnn, inn, br, bi = args
    return get_nonlinearity(kw["name"])(rnn @ h + inn @ x + br + bi) - h

def rnn_sim(h0, *args, **kwargs):
    kw = {"name": "tanh"}
    kw.update(kwargs)

    x, rnn, inn, br, bi = args
    h = get_nonlinearity(kw["name"])(rnn @ h0 + inn @ x + br + bi)
    return h

def jacobian(h, *args, **kwargs):
    '''Jacobian of the RNN.'''
    kw = {"name": "tanh"}
    kw.update(kwargs)

    x, rnn, inn, br, bi = args
    inner = rnn @ h + inn @ x  + br + bi
    if kw["name"] == "tanh":
        return np.diag((1 - pow(np.tanh(inner), 2))) @ rnn
    elif kw["name"] == "relu":
        return np.dot(np.diag(drelu(inner)), rnn)
    elif kw["name"] == "clip(tanh)":
        dtanh_Ax = np.diag((1 - pow(np.tanh(inner), 2)))
        dclip_tanh_Ax = dclip(np.tanh(inner))
        return dtanh_Ax @ rnn * dclip_tanh_Ax

def jacobian_rhs(h, *args):
    '''Jacobian of the rhs of the RNN.'''
    return jacobian(h, args) - np.identity(64)

def check_fixed_point(fp):
    '''Check if fixed point is between -1 and 1 (because of tanh).'''
    return np.all([abs(c) <= 1 for c in fp])

def get_fixed_points_by_sim(x, rnn, inn, br, bi, rp=100, dim_h=8, name="tanh"):
    '''Obtain fixed points, with convergence values starting from #rp random initial values.'''
    def rhs_wrapper(x, *args): return rhs(x, *args, name=name)

    fps = []
    stop_stim = False
    for r in range(rp):
        h_0 = np.random.uniform(low=-1, high=1, size=dim_h)
        if not stop_stim:
            for _ in range(10): h_0 = rnn_sim(h_0, x, rnn, inn, br, bi, name=name)
        fp, infodic, ier = fsolve(rhs_wrapper, h_0, args=(x, rnn, inn, br, bi), xtol=1e-15, full_output=True)[:3]

        flag = (abs(np.mean(infodic["fvec"])) <= 1e-15) and ier
        for ref in fps:
            if not different(fp, ref)[0]: flag = False; break
        if flag: fps.append(fp)

        if r > rp//2: stop_stim = True

    if name == "tanh":
        if not np.all([check_fixed_point(fp) for fp in fps]): raise ValueError("fixed point is not between -1 and 1!")
    return fps

def get_unique_fixed_points(fps):
    '''Obtain unique fixed points.'''
    unique_fps = []
    for fp in fps:
        flag = True
        for ref in unique_fps:
            if not different(fp, ref)[0]: flag = False; break
        if flag: unique_fps.append(fp)

    if not np.all([check_fixed_point(fp) for fp in unique_fps]): raise ValueError("fixed point is not between -1 and 1!")
    return unique_fps

def get_jacobians(x, rnn, inn, br, bi):
    '''Obtain Jacobian of all the fixed points.'''
    fps = get_fixed_points_by_sim(x, rnn, inn, br, bi)
    if len(fps) > 1: print(f"{x} obtained {len(fps)} fixed points!")
    return [jacobian(fp, [x, rnn, inn, br, bi]) for fp in fps]

def get_stability(fp, *args, **kwargs):
    kw = {"name": "tanh"}
    kw.update(kwargs)

    J = jacobian(fp, *args, name=kw["name"])
    evs = np.linalg.eigvals(J)
    stable = np.all(abs(evs) < 1)
    return stable

def get_sorted_eig(fp, args):
    J = jacobian(fp, args)
    evs, ews = np.linalg.eig(J)
    idx = np.flip(np.argsort(abs(evs)))
    evs_sorted = evs[idx]
    ews_sorted = ews.T[idx]
    return evs_sorted, ews_sorted # rows = eigenvectors

########################################################
#                         GRU                          #
########################################################

sigmoid = lambda x: 1/(1 + np.exp(-x))

def gru_sim(v, *args):
    x, dim_h, Wr, Wz, Wh, Ur, Uz, Uh, bhr, bhz, bhh, bir, biz, bih = args
    z, r, hh, h = v[:dim_h], v[dim_h:2*dim_h], v[2*dim_h:3*dim_h], v[3*dim_h:]
    
    f1 = sigmoid(Ur @ x + bir + Wr @ h + bhr)
    f2 = sigmoid(Uz @ x + biz + Wz @ h + bhz)
    f3 = np.tanh(Uh @ x + bih + r*(Wh @ h + bhh))
    f4 = (1 - z)*hh + z*h
    res = np.array([f1, f2, f3, f4]).reshape(-1)
    return res

def rhs_gru(v, *args):
    """
    Solves:
        r = sigmoid(Wr @ x + bir + Ur @ h + bhr)
        z = sigmoid(Wz @ x + biz + Uz @ h + bhz)
        h = tanh(Wh @ x + bih + r*(Uh @ h + bhh)) - hh
        h = (1-z)*hh + z*h
    """
    x, dim_h, Wr, Wz, Wh, Ur, Uz, Uh, bhr, bhz, bhh, bir, biz, bih = args
    z, r, hh, h = v[:dim_h], v[dim_h:2*dim_h], v[2*dim_h:3*dim_h], v[3*dim_h:]
    
    f1 = sigmoid(Ur @ x + bir + Wr @ h + bhr) - r
    f2 = sigmoid(Uz @ x + biz + Wz @ h + bhz) - z
    f3 = np.tanh(Uh @ x + bih + r*(Wh @ h + bhh)) - hh
    f4 = (1 - z)*hh + z*h - h
    res = np.array([f1, f2, f3, f4]).reshape(-1)
    return res

def get_fixed_points_by_sim_gru(x, gru, rp=100, dim_h=8):
    '''Obtain fixed points, with #rp random initial values.'''
    gru.weight_hh.requires_grad = False
    gru.weight_ih.requires_grad = False
    gru.bias_hh.requires_grad = False
    gru.bias_ih.requires_grad = False
    args = [x, dim_h, *np.split(gru.weight_hh.numpy(), 3), *np.split(gru.weight_ih.numpy(), 3),
            *np.split(gru.bias_hh.numpy(), 3), *np.split(gru.bias_ih.numpy(), 3)]
    def rhs_wrapper(x, args): return rhs_gru(x, *args)

    fps = []
    for _ in range(rp):
        v0 = torch.rand(dim_h*4).numpy()
        fp, infodic, ier = fsolve(rhs_wrapper, v0, args=args, xtol=1e-15, full_output=True)[:3]
        # fprime=jacobian_rhs

        flag = (abs(np.mean(infodic["fvec"])) <= 1e-15) and ier
        for ref in fps:
            if not different(fp, ref)[0]: flag = False; break
        if flag: fps.append(fp)

    if not np.all([check_fixed_point(fp) for fp in fps]): raise ValueError("fixed point is not between -1 and 1!")
    return fps
