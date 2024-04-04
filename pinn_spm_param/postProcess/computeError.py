import json
import os
import sys

import numpy as np

sys.path.append("../util")
from pathlib import Path

import argument
import tensorflow as tf
from keras import layers, regularizers
from myNN import *
from keras.backend import set_floatx

set_floatx("float64")

from forwardPass import (
    from_param_list_to_str,
    make_data_dict,
    make_var_params_from_data,
    pinn_pred,
    pinn_pred_struct,
)
from init_pinn import initialize_nn_from_params_config, safe_load


def computeError(dataDict, predDict, debug=False):
    phie_rescaled = predDict["phie"]
    phis_c_rescaled = predDict["phis_c"]
    cs_a_rescaled = predDict["cs_a"]
    cs_c_rescaled = predDict["cs_c"]

    yTest_phie = dataDict["phie"]
    yTest_phis_c = dataDict["phis_c"]
    yTest_cs_a = dataDict["cs_a"]
    yTest_cs_c = dataDict["cs_c"]

    if debug:
        import matplotlib.pyplot as plt

    globalError = 0
    tmp = np.mean(
        abs(yTest_phie - phie_rescaled)
        / np.clip(abs(yTest_phie), a_min=1e-16, a_max=None)
    )
    globalError += tmp
    globalError_phie = tmp
    if debug:
        plt.plot(yTest_phie, phie_rescaled, "o")
        plt.title(f"phie {tmp}")
        plt.show()
    tmp = np.mean(
        abs(yTest_phis_c - phis_c_rescaled)
        / np.clip(abs(yTest_phis_c), a_min=1e-16, a_max=None)
    )
    globalError += tmp
    globalError_phis_c = tmp
    if debug:
        plt.plot(yTest_phis_c, phis_c_rescaled, "o")
        plt.title(f"phis_c {tmp}")
        plt.show()
    tmp = np.mean(
        abs(yTest_cs_a - cs_a_rescaled)
        / np.clip(abs(yTest_cs_a), a_min=1e-16, a_max=None)
    )
    globalError += tmp
    globalError_cs_a = tmp
    if debug:
        plt.plot(yTest_cs_a, cs_a_rescaled, "o")
        plt.title(f"cs_a {tmp}")
        plt.show()
    tmp = np.mean(
        abs(yTest_cs_c - cs_c_rescaled)
        / np.clip(abs(yTest_cs_c), a_min=1e-16, a_max=None)
    )
    globalError += tmp
    globalError_cs_c = tmp
    if debug:
        plt.plot(yTest_cs_c, cs_c_rescaled, "o")
        plt.title(f"cs_c {tmp}")
        plt.show()

    return globalError, {
        "phie": globalError_phie,
        "phis_c": globalError_phis_c,
        "cs_a": globalError_cs_a,
        "cs_c": globalError_cs_c,
    }


def init_error(
    modelFolder,
    dataFolder,
    params,
    nn=None,
    data_dict=None,
    var_dict=None,
    params_dict=None,
    params_list=None,
):
    print("INFO: Using modelFolder : ", modelFolder)
    print("INFO: Using dataFolder : ", dataFolder)
    print("INFO: Loading from config file")

    if not nn:
        with open(os.path.join(modelFolder, "config.json")) as json_file:
            configDict = json.load(json_file)
            nn = initialize_nn_from_params_config(params, configDict)
    if not data_dict:
        data_dict = make_data_dict(dataFolder, params_list)
    if not var_dict or not params_dict:
        var_dict, params_dict = make_var_params_from_data(nn, data_dict)
    nn = safe_load(nn, os.path.join(modelFolder, "best.weights.h5"))

    return nn, data_dict, var_dict, params_dict


if __name__ == "__main__":
    # Read command line arguments
    args = argument.initArg()
    if args.params_list is None:
        params_list = None
    else:
        params_list = [float(param_val) for param_val in args.params_list]

    modelFolder = args.modelFolder
    dataFolder = args.dataFolder
    if args.simpleModel:
        from spm_simpler import makeParams
    else:
        from spm import makeParams
    params = makeParams()

    nn, data_dict, var_dict, params_dict = init_error(
        modelFolder, dataFolder, params, params_list=params_list
    )
    pred_dict = pinn_pred(nn, var_dict, params_dict)
    globalError, _ = computeError(data_dict, pred_dict)
    print(globalError)
