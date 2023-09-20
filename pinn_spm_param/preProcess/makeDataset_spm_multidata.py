import os
import sys

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split

sys.path.append("../util")
import argument

# Read command line arguments
args = argument.initArg()

if args.simpleModel:
    from spm_simpler import *
else:
    from spm import *

params = makeParams()

dataFolder = args.dataFolder
freqT = args.frequencyDownsamplingT
dynamicFreqT = False
if freqT == 1:
    dynamicFreqT = True
    target_nt = args.n_t

input_params = False
if not args.params_list is None:
    input_params = True
if input_params:
    deg_i0_a = float(args.params_list[0])
    deg_ds_c = float(args.params_list[1])
else:
    deg_i0_a = 1.0
    deg_ds_c = 1.0


use_multi_data = False
if not args.data_list is None:
    use_multi_data = True
if use_multi_data:
    multi_deg_i0_a = [
        float(args.params_list[2 * i])
        for i in range(len(args.params_list) // 2)
    ]
    multi_deg_ds_c = [
        float(args.params_list[2 * i + 1])
        for i in range(len(args.params_list) // 2)
    ]

data_list = args.data_list


n_data = len(data_list)

# load data
multi_data = [
    np.load(os.path.join(dataFolder, data_file)) for data_file in data_list
]


for idata, (data, deg_i0_a, deg_ds_c) in enumerate(
    zip(multi_data, multi_deg_i0_a, multi_deg_ds_c)
):
    if dynamicFreqT:
        n_t = data["t"].shape[0]
        freqT = n_t // target_nt

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
    x_phie = t_phie.astype("float64")
    x_params_phie = np.zeros((x_phie.shape[0], 2))
    x_params_phie[:, 0] = deg_i0_a
    x_params_phie[:, 1] = deg_ds_c
    y_phie = np.reshape(phie, (len(phie), 1)).astype("float64")

    if idata == 0:
        x_phie_tot = x_phie
        x_params_phie_tot = x_params_phie
        y_phie_tot = y_phie
    else:
        x_phie_tot = np.vstack((x_phie_tot, x_phie))
        x_params_phie_tot = np.vstack((x_params_phie_tot, x_params_phie))
        y_phie_tot = np.vstack((y_phie_tot, y_phie))

    if idata == len(multi_data) - 1:
        (
            x_train,
            x_test,
            y_train,
            y_test,
            x_params_train,
            x_params_test,
        ) = train_test_split(
            x_phie_tot,
            y_phie_tot,
            x_params_phie_tot,
            test_size=0.1,
            random_state=42,
        )
        np.savez(
            os.path.join(dataFolder, "data_phie_multi.npz"),
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
    x_phis_c = t_phis_c.astype("float64")
    x_params_phis_c = np.zeros((x_phis_c.shape[0], 2))
    x_params_phis_c[:, 0] = deg_i0_a
    x_params_phis_c[:, 1] = deg_ds_c
    y_phis_c = np.reshape(phis_c, (len(phis_c), 1)).astype("float64")

    if idata == 0:
        x_phis_c_tot = x_phis_c
        x_params_phis_c_tot = x_params_phis_c
        y_phis_c_tot = y_phis_c
    else:
        x_phis_c_tot = np.vstack((x_phis_c_tot, x_phis_c))
        x_params_phis_c_tot = np.vstack((x_params_phis_c_tot, x_params_phis_c))
        y_phis_c_tot = np.vstack((y_phis_c_tot, y_phis_c))

    if idata == len(multi_data) - 1:
        (
            x_train,
            x_test,
            y_train,
            y_test,
            x_params_train,
            x_params_test,
        ) = train_test_split(
            x_phis_c_tot,
            y_phis_c_tot,
            x_params_phis_c_tot,
            test_size=0.1,
            random_state=42,
        )
        np.savez(
            os.path.join(dataFolder, "data_phis_c_multi.npz"),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_params_train=x_params_train,
            x_params_test=x_params_test,
        )

    # cs anode
    cs_a = np.reshape(cs_a, (n_t * n_r, 1))
    x_cs_a = np.hstack((t_2d_flatten, r_a_2d_flatten)).astype("float64")
    x_params_cs_a = np.zeros((x_cs_a.shape[0], 2))
    x_params_cs_a[:, 0] = deg_i0_a
    x_params_cs_a[:, 1] = deg_ds_c
    y_cs_a = np.reshape(cs_a, (len(cs_a), 1)).astype("float64")

    if idata == 0:
        x_cs_a_tot = x_cs_a
        x_params_cs_a_tot = x_params_cs_a
        y_cs_a_tot = y_cs_a
    else:
        x_cs_a_tot = np.vstack((x_cs_a_tot, x_cs_a))
        x_params_cs_a_tot = np.vstack((x_params_cs_a_tot, x_params_cs_a))
        y_cs_a_tot = np.vstack((y_cs_a_tot, y_cs_a))

    if idata == len(multi_data) - 1:
        (
            x_train,
            x_test,
            y_train,
            y_test,
            x_params_train,
            x_params_test,
        ) = train_test_split(
            x_cs_a_tot,
            y_cs_a_tot,
            x_params_cs_a_tot,
            test_size=0.1,
            random_state=42,
        )
        np.savez(
            os.path.join(dataFolder, "data_cs_a_multi.npz"),
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
    x_cs_c = np.hstack((t_2d_flatten, r_c_2d_flatten)).astype("float64")
    x_params_cs_c = np.zeros((x_cs_c.shape[0], 2))
    x_params_cs_c[:, 0] = deg_i0_a
    x_params_cs_c[:, 1] = deg_ds_c
    y_cs_c = np.reshape(cs_c, (len(cs_c), 1)).astype("float64")

    if idata == 0:
        x_cs_c_tot = x_cs_c
        x_params_cs_c_tot = x_params_cs_c
        y_cs_c_tot = y_cs_c
    else:
        x_cs_c_tot = np.vstack((x_cs_c_tot, x_cs_c))
        x_params_cs_c_tot = np.vstack((x_params_cs_c_tot, x_params_cs_c))
        y_cs_c_tot = np.vstack((y_cs_c_tot, y_cs_c))
    if idata == len(multi_data) - 1:
        (
            x_train,
            x_test,
            y_train,
            y_test,
            x_params_train,
            x_params_test,
        ) = train_test_split(
            x_cs_c_tot,
            y_cs_c_tot,
            x_params_cs_c_tot,
            test_size=0.1,
            random_state=42,
        )
        np.savez(
            os.path.join(dataFolder, "data_cs_c_multi.npz"),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_params_train=x_params_train,
            x_params_test=x_params_test,
        )
