import argparse
import json
import os
import sys

import numpy as np

parser = argparse.ArgumentParser(description="Bayes cal preprocess")
parser.add_argument(
    "-uf",
    "--utilFolder",
    type=str,
    metavar="",
    required=True,
    help="util folder of model",
    default=None,
)
parser.add_argument(
    "-n_t",
    "--n_t",
    type=int,
    metavar="",
    required=False,
    help="number of measurements",
    default=100,
)
parser.add_argument(
    "-noise",
    "--noise",
    type=float,
    metavar="",
    required=False,
    help="noise level",
    default=0,
)

args, unknown = parser.parse_known_args()
nT_target = args.n_t
noise = args.noise

sys.path.append(args.utilFolder)

import argument
import tensorflow as tf
from keras import layers, regularizers
from keras.backend import set_floatx
from myNN import *

set_floatx("float64")

# Read command line arguments
args_spm = argument.initArg()

import corner  # plotting package
import scipy.optimize as opt
from forwardPass import (
    from_param_list_to_str,
    pinn_pred_phis_c,
    rescale_param_list,
)
from init_pinn import initialize_nn_from_params_config, safe_load
from matplotlib import rc, rcParams
from prettyPlot.plotting import plt, pretty_labels

if args_spm.simpleModel:
    from spm_simpler import *
else:
    from spm import *


filename = f"dataMeasured_{nT_target}_{noise:.2g}.npz"
data_phis_c = np.load(filename)["data"].astype("float64")
data_t = np.load(filename)["t"].astype("float64")
deg_params_obs = np.load(filename)["deg_params"].astype("float64")

n_t = data_t.shape[0]
inpt_grid = np.zeros((n_t, 1)).astype("float64")
inpt_grid[:, 0] = data_t

params = makeParams()
params["deg_params_names"] = ["i0_a", "ds_c"]
params["n_params"] = 2
modelFolder = args_spm.modelFolder
print("INFO: Using modelFolder : ", modelFolder)

if not os.path.exists(os.path.join(modelFolder, "config.json")):
    print("Looking for file ", os.path.join(modelFolder, "config.json"))
    sys.exit("ERROR: Config file could not be found necessary")
else:
    print("INFO: Loading from config file")
    with open(os.path.join(modelFolder, "config.json")) as json_file:
        configDict = json.load(json_file)
    nn = initialize_nn_from_params_config(params, configDict)
    activeInt = configDict["activeInt"]
    activeBound = configDict["activeBound"]
    activeData = configDict["activeData"]
    activeReg = configDict["activeReg"]
    try:
        params_min = configDict["params_min"]
    except:
        params_min = [
            params["deg_" + par_name + "_min"]
            for par_name in params["deg_params_names"]
        ]
    try:
        params_max = configDict["params_max"]
    except:
        params_max = [
            params["deg_" + par_name + "_max"]
            for par_name in params["deg_params_names"]
        ]

nn = safe_load(nn, os.path.join(modelFolder, "best.weights.h5"))
model = nn.model
# rescale
resc_r = params["rescale_R"]
resc_t = params["rescale_T"]


def forwardModel(unknowns, inpt):
    t = np.reshape(inpt, (inpt.shape[0], 1))
    dummyR = params["Rs_c"] * np.ones((inpt.shape[0], 1))
    var_dict = {}
    params_dict = {}

    var_dict["phis_c"] = [t / resc_t]
    var_dict["phis_c_unr"] = [t]
    var_dict["phis_c_full"] = [t / resc_t, dummyR / resc_r]
    var_dict["phis_c_unr_full"] = [t, dummyR]

    params_dict["phis_c_unr"] = [
        np.clip(
            unknowns[i] * np.ones((len(t), 1)).astype("float64"),
            nn.params_min[i],
            nn.params_max[i],
        )
        for i in range(len(unknowns))
    ]
    params_dict["phis_c"] = rescale_param_list(nn, params_dict["phis_c_unr"])

    pred_dict = pinn_pred_phis_c(nn, var_dict, params_dict)

    phis_c_rescaled = pred_dict["phis_c"]

    return phis_c_rescaled[:, 0]


def hypercube_combinations(val_list):
    if val_list:
        for el in val_list[0]:
            for combination in hypercube_combinations(val_list[1:]):
                yield [el] + combination
    else:
        yield []


verts = [[0, 1] for _ in range(params["n_params"])]
combs = hypercube_combinations(verts)


dataFolder = args_spm.dataFolder
figureFolder = "Figures"
os.makedirs(figureFolder, exist_ok=True)


fig = plt.figure()
par_list = list(deg_params_obs)
param_string = from_param_list_to_str(par_list)
solData = np.load(os.path.join(dataFolder, f"solution{param_string}.npz"))
plt.plot(
    solData["t"],
    solData["phis_c"],
    linewidth=3,
    color="r",
)
for comb in list(combs):
    par_list = []
    for ipar, name in enumerate(params["deg_params_names"]):
        if comb[ipar] == 0:
            par_list.append(params["deg_" + name + "_min"])
        elif comb[ipar] == 1:
            par_list.append(params["deg_" + name + "_max"])
    plt.plot(data_t, forwardModel(par_list, data_t), "--", color="k")
pretty_labels("time [s]", r"$\phi_{s,+}$ [V]", 14, title="all pred")
plt.savefig(os.path.join(figureFolder, "bounding.png"))
plt.close()


par_list = list(deg_params_obs)
param_string = from_param_list_to_str(par_list)
solData = np.load(os.path.join(dataFolder, f"solution{param_string}.npz"))
fig = plt.figure()
plt.plot(
    solData["t"],
    abs(forwardModel(par_list, solData["t"]) - solData["phis_c"])
    / solData["phis_c"],
    linewidth=3,
    color="r",
)
combs = hypercube_combinations(verts)
for comb in combs:
    par_list = []
    for ipar, name in enumerate(params["deg_params_names"]):
        if comb[ipar] == 0:
            par_list.append(params["deg_" + name + "_min"])
        elif comb[ipar] == 1:
            par_list.append(params["deg_" + name + "_max"])
    param_string = from_param_list_to_str(par_list)
    solData = np.load(os.path.join(dataFolder, f"solution{param_string}.npz"))
    plt.plot(
        solData["t"],
        abs(forwardModel(par_list, solData["t"]) - solData["phis_c"])
        / solData["phis_c"],
        linewidth=3,
        color="k",
    )
pretty_labels("time [s]", r"Rel. Err. $\phi_{s,+}$", 14, title="all pred")
plt.savefig(os.path.join(figureFolder, "err.png"))
plt.close()
