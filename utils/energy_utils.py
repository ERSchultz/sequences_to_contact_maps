import numpy as np


def calculate_E_S(x, chi):
    if x is None or chi is None:
        return None, None
    s = calculate_S(x, chi)
    e = s_to_E(s)
    return e, s

def calculate_E(x, chi):
    s = calculate_S(x, chi)
    e = s_to_E(s)
    return e

def s_to_E(s):
    if s is None:
        return None
        
    return s + s.T - np.diag(np.diagonal(s).copy())

def calculate_S(x, chi):
    # zero lower triangle (double check)
    chi = np.triu(chi)

    try:
        s = x @ chi @ x.T
    except ValueError as e:
        print('x', x, x.shape)
        print('chi', chi, chi.shape)
        raise
    return s
