import json
import os
import sys

import numpy as np

sys.path.append("../util")
import argument
import tensorflow as tf
from keras import layers, regularizers
from keras.backend import set_floatx
from myNN import *
from plotsUtil_batt import *

set_floatx("float64")

from init_pinn import initialize_nn_from_params_config, safe_load

print("\n\nINFO: PLOT CORRELATION PINN DATA\n\n")


def log(inpt, name):
    if optimized == False:
        print(name)
        print("min %.2f max %.2f " % (np.amin(inpt), np.amax(inpt)))


def corr_plot(args):
    if args.simpleModel:
        from spm_simpler import makeParams
    else:
        from spm import makeParams

    params = makeParams()

    if args.params_list is None:
        params_list = None
    else:
        params_list = [float(param_val) for param_val in args.params_list]

    if not args.verbose:
        import matplotlib

        matplotlib.use("Agg")

    modelFolder = args.modelFolder
    print("INFO: Using modelFolder : ", modelFolder)
    dataFolder = args.dataFolder

    if not os.path.exists(os.path.join(modelFolder, "config.json")):
        print("Looking for file ", os.path.join(modelFolder, "config.json"))
        sys.exit("ERROR: Config file could not be found necessary")
    else:
        print("INFO: Loading from config file")
        with open(os.path.join(modelFolder, "config.json")) as json_file:
            configDict = json.load(json_file)
        nn = initialize_nn_from_params_config(params, configDict)

    nn = safe_load(nn, os.path.join(modelFolder, "best.weights.h5"))
    model = nn.model
    from forwardPass import (
        from_param_list_to_str,
        make_data_dict,
        make_var_params_from_data,
        pinn_pred,
    )

    data_dict = make_data_dict(dataFolder, params_list)
    var_dict, params_dict = make_var_params_from_data(nn, data_dict)
    pred_dict = pinn_pred(nn, var_dict, params_dict)

    figureFolder = "Figures"
    modelFolderFig = (
        modelFolder.replace("../", "")
        .replace("/Log", "")
        .replace("/Model", "")
        .replace("Log", "")
        .replace("Model", "")
        .replace("/", "_")
        .replace(".", "")
    )

    if not args.verbose:
        os.makedirs(figureFolder, exist_ok=True)
        os.makedirs(os.path.join(figureFolder, modelFolderFig), exist_ok=True)

    yTest_phie = data_dict["phie"]
    yTest_phis_c = data_dict["phis_c"]
    yTest_cs_a = data_dict["cs_a"]
    yTest_cs_c = data_dict["cs_c"]

    phie_rescaled = pred_dict["phie"]
    phis_c_rescaled = pred_dict["phis_c"]
    cs_a_rescaled = pred_dict["cs_a"]
    cs_c_rescaled = pred_dict["cs_c"]

    fig, axs = plt.subplots(2, 2, figsize=(2 * 3, 2 * 3))
    axs[0, 0].plot(yTest_phie, phie_rescaled, "o", color="k")
    axs[0, 0].plot(yTest_phie, yTest_phie, "--", color="k")
    pretty_labels(r"$\phi_{e,PDE}$", r"$\phi_{e,PINN}$", 14, ax=axs[0, 0])
    axs[0, 1].plot(yTest_phis_c, phis_c_rescaled, "o", color="k")
    axs[0, 1].plot(yTest_phis_c, yTest_phis_c, "--", color="k")
    pretty_labels(r"$\phi_{s,c,PDE}$", r"$\phi_{s,c,PINN}$", 14, ax=axs[0, 1])
    axs[1, 0].plot(yTest_cs_a, cs_a_rescaled, "o", color="k")
    axs[1, 0].plot(yTest_cs_a, yTest_cs_a, "--", color="k")
    pretty_labels(r"$c_{s,a,PDE}$", r"$c_{s,a,PINN}$", 14, ax=axs[1, 0])
    axs[1, 1].plot(yTest_cs_c, cs_c_rescaled, "o", color="k")
    axs[1, 1].plot(yTest_cs_c, yTest_cs_c, "--", color="k")
    pretty_labels(r"$c_{s,c,PDE}$", r"$c_{s,c,PINN}$", 14, ax=axs[1, 1])
    currentAxis = 2

    if not args.verbose:
        param_str = from_param_list_to_str(params_list)
        plt.savefig(
            os.path.join(
                figureFolder, modelFolderFig, f"corrDataPINN{param_str}"
            )
            + ".png"
        )

    if args.verbose:
        plt.show()


if __name__ == "__main__":
    # Read command line arguments
    args = argument.initArg()
    corr_plot(args)
