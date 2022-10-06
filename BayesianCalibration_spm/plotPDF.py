import os
import sys

import numpy as np

sys.path.append("../pinn_spm_param/util")
import argument
import tensorflow as tf
from myNN import *
from plotsUtil import *
from tensorflow import keras
from tensorflow.keras import layers, regularizers

tf.keras.backend.set_floatx("float64")

# Read command line arguments
args = argument.initArg()

import corner  # plotting package
import emcee  # montecarlo sampler
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib import rc, rcParams
from plotsUtil import *
from spm_simpler import *

flat_samples = np.load("result_exact_100.npy")

labels = [r"$i0_a$", r"$Ds_c$"]

import corner

# activate latex text rendering
rc("font", weight="bold", family="Times New Roman", size=14)
fig = corner.corner(
    flat_samples,
    range=[(0.5, 4), (1, 10)],
    plot_datapoints=False,
    bins=300,
    smooth=1,
    smooth1d=1,
    labels=labels,
    truths=[2, 2],
    use_math_text=True,
    label_kwargs={
        "fontsize": 14,
        "fontweight": "bold",
        "fontname": "Times New Roman",
    },
    title_kwargs={
        "fontsize": 14,
        "fontweight": "bold",
        "fontname": "Times New Roman",
    },
)

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
ndim = 2
for i in range(ndim):
    ax = axes[i]
    ax.plot(flat_samples[:, i], "k", alpha=0.3)
    ax.set_xlim(0, len(flat_samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")

plt.show()
