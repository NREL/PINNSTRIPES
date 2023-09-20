import numpy as np


def gradient(state, variable, axisVar):
    return np.gradient(state, variable, axis=axisVar, edge_order=2)
