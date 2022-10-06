import os
import sys

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split

sys.path.append("../util")
import argument

# Read command line arguments
args = argument.initArg()
from spm_simpler import *

params = makeParams()

dataFolder = args.dataFolder
freqT = args.frequencyDownsamplingT

input_params = False
if not args.params_list is None:
    input_params = True
if input_params:
    deg_i0_a = float(args.params_list[0])
    deg_ds_c = float(args.params_list[1])
else:
    deg_i0_a = 1.0
    deg_ds_c = 1.0


# load data, from Raissi et. al
if input_params:
    data = np.load(
        os.path.join(
            dataFolder, "solution_%.2g_%.2g.npz" % (deg_i0_a, deg_ds_c)
        )
    )
else:
    data = np.load(os.path.join(dataFolder, "solution.npz"))
t = data["t"][::freqT].astype("float64")
r_a = data["r_a"].astype("float64")
r_c = data["r_c"].astype("float64")

# state
phie = data["phie"][::freqT].astype("float64")
phis_c = data["phis_c"][::freqT].astype("float64")
cs_a = data["cs_a"][::freqT, :].astype("float64")
cs_c = data["cs_c"][::freqT, :].astype("float64")

n_r = len(r_a)
n_t = len(t)

t_2d = np.reshape(t, (n_t, 1))
r_a_2d = np.reshape(r_a, (1, n_r))
r_c_2d = np.reshape(r_c, (1, n_r))
t_2d = np.repeat(t_2d, n_r, axis=1)
r_a_2d = np.repeat(r_a_2d, n_t, axis=0)
r_c_2d = np.repeat(r_c_2d, n_t, axis=0)
t_1d = np.reshape(t, (n_t, 1))
t_2d_flatten = np.reshape(t_2d, (n_r * n_t, 1))
r_a_2d_flatten = np.reshape(r_a_2d, (n_r * n_t, 1))
r_c_2d_flatten = np.reshape(r_c_2d, (n_r * n_t, 1))


# Phie
t_phie = t_1d
# Assemble
x = t_phie.astype("float64")
x_params = np.zeros((x.shape[0], 2))
x_params[:, 0] = deg_i0_a
x_params[:, 1] = deg_ds_c
y = np.reshape(phie, (len(phie), 1)).astype("float64")
(
    x_train,
    x_test,
    y_train,
    y_test,
    x_params_train,
    x_params_test,
) = train_test_split(x, y, x_params, test_size=0.1, random_state=42)
if input_params:
    np.savez(
        os.path.join(
            dataFolder, "data_phie_%.2g_%.2g.npz" % (deg_i0_a, deg_ds_c)
        ),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_params_train=x_params_train,
        x_params_test=x_params_test,
    )
else:
    np.savez(
        os.path.join(dataFolder, "data_phie.npz"),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_params_train=x_params_train,
        x_params_test=x_params_test,
    )
# Phis_c
t_phis_c = t_1d
# Assemble
x = t_phis_c.astype("float64")
x_params = np.zeros((x.shape[0], 2))
x_params[:, 0] = deg_i0_a
x_params[:, 1] = deg_ds_c
y = np.reshape(phis_c, (len(phis_c), 1)).astype("float64")
(
    x_train,
    x_test,
    y_train,
    y_test,
    x_params_train,
    x_params_test,
) = train_test_split(x, y, x_params, test_size=0.1, random_state=42)
if input_params:
    np.savez(
        os.path.join(
            dataFolder, "data_phis_c_%.2g_%.2g.npz" % (deg_i0_a, deg_ds_c)
        ),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_params_train=x_params_train,
        x_params_test=x_params_test,
    )
else:
    np.savez(
        os.path.join(dataFolder, "data_phis_c.npz"),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_params_train=x_params_train,
        x_params_test=x_params_test,
    )


# cs anode
cs_a = np.reshape(cs_a, (n_t * n_r, 1))
# Assemble
x = np.hstack((t_2d_flatten, r_a_2d_flatten)).astype("float64")
x_params = np.zeros((x.shape[0], 2))
x_params[:, 0] = deg_i0_a
x_params[:, 1] = deg_ds_c
y = np.reshape(cs_a, (len(cs_a), 1)).astype("float64")
(
    x_train,
    x_test,
    y_train,
    y_test,
    x_params_train,
    x_params_test,
) = train_test_split(x, y, x_params, test_size=0.01, random_state=42)
if input_params:
    np.savez(
        os.path.join(
            dataFolder, "data_cs_a_%.2g_%.2g.npz" % (deg_i0_a, deg_ds_c)
        ),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_params_train=x_params_train,
        x_params_test=x_params_test,
    )
else:
    np.savez(
        os.path.join(dataFolder, "data_cs_a.npz"),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_params_train=x_params_train,
        x_params_test=x_params_test,
    )

# cs cathode
cs_c = np.reshape(cs_c, (n_t * n_r, 1))
# Assemble
x = np.hstack((t_2d_flatten, r_c_2d_flatten)).astype("float64")
x_params = np.zeros((x.shape[0], 2))
x_params[:, 0] = deg_i0_a
x_params[:, 1] = deg_ds_c
y = np.reshape(cs_c, (len(cs_c), 1)).astype("float64")
(
    x_train,
    x_test,
    y_train,
    y_test,
    x_params_train,
    x_params_test,
) = train_test_split(x, y, x_params, test_size=0.01, random_state=42)
if input_params:
    np.savez(
        os.path.join(
            dataFolder, "data_cs_c_%.2g_%.2g.npz" % (deg_i0_a, deg_ds_c)
        ),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_params_train=x_params_train,
        x_params_test=x_params_test,
    )
else:
    np.savez(
        os.path.join(dataFolder, "data_cs_c.npz"),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_params_train=x_params_train,
        x_params_test=x_params_test,
    )
