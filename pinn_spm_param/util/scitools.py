import numpy as np


def gradient(state, variable, axisVar):
    # stepSize = np.mean(np.diff(variable))
    return np.gradient(state, variable, axis=axisVar, edge_order=2)
