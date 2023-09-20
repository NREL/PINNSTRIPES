import json
import os
import sys

import numpy as np

sys.path.append("../util")
from pathlib import Path

import argument
import tensorflow as tf
from myNN import *
from plotsUtil import *
from plotsUtil_batt import *
from tensorflow import keras
from tensorflow.keras import layers, regularizers

tf.keras.backend.set_floatx("float64")

from forwardPass import (
    from_param_list_to_str,
    make_data_dict,
    make_data_dict_struct,
    make_var_params_from_data,
    pinn_pred,
    pinn_pred_struct,
)
from init_pinn import initialize_nn_from_params_config, safe_load

print("\n\nINFO: PLOTTING MOVIE OF THE PINN TRAINING\n\n")


def getModelStep(modelFolder):
    # Read Time
    model_tmp = os.listdir(modelFolder)
    # remove non floats
    for i, entry in reversed(list(enumerate(model_tmp))):
        if not entry.startswith("step_"):
            a = model_tmp.pop(i)
        else:
            try:
                a = float(entry[5:-3])
            except ValueError:
                a = model_tmp.pop(i)

    step_float = [float(entry[5:-3]) for entry in model_tmp]
    step_str = [entry[5:-3] for entry in model_tmp]
    index_sort = np.argsort(step_float)
    step_float_sorted = [step_float[i] for i in list(index_sort)]
    step_str_sorted = [step_str[i] for i in list(index_sort)]
    if len(step_str_sorted) == 0:
        sys.exit("ERROR: no weight saved")

    return step_float_sorted, step_str_sorted


def makePlot(movieDir, solDict, params_list, indMov, stepID, data_dict=None):
    figureFolder = movieDir
    path = Path(figureFolder)
    path.mkdir(parents=True, exist_ok=True)

    timeMiddle = 200

    t_test = solDict["t_test"]
    tmax = solDict["tmax"]
    r_test_a = solDict["r_test_a"]
    r_test_c = solDict["r_test_c"]
    cs_a = solDict["cs_a"]
    cs_c = solDict["cs_c"]
    phie = solDict["phie"]
    phis_c = solDict["phis_c"]

    string_params = from_param_list_to_str(params_list)
    # Plot result battery style
    # CS A
    file_path_name = os.path.join(
        figureFolder, f"cs_a{string_params}_{indMov}.png"
    )
    try:
        temp_dat = data_dict["cs_a_t"]
        spac_dat = data_dict["cs_a_r"]
        field_dat = data_dict["cs_a"]
    except:
        temp_dat = None
        spac_dat = None
        field_dat = None

    line_cs_results(
        temp_pred=t_test[:, 0, 0, 0],
        spac_pred=r_test_a[0, :, 0, 0],
        field_pred=cs_a[:, :, 0, 0],
        time_stamps=[0, 200, 400],
        xlabel=r"r ($\mu$m)",
        ylabel=r"C$_{s,Li,an}$ (kmol/m$^3$)",
        title=f"Step {stepID}",
        file_path_name=file_path_name,
        temp_dat=temp_dat,
        spac_dat=spac_dat,
        field_dat=field_dat,
        verbose=False,
    )
    # CS C
    file_path_name = os.path.join(
        figureFolder, f"cs_c{string_params}_{indMov}.png"
    )
    try:
        temp_dat = data_dict["cs_c_t"]
        spac_dat = data_dict["cs_c_r"]
        field_dat = data_dict["cs_c"]
    except:
        temp_dat = None
        spac_dat = None
        field_dat = None

    line_cs_results(
        temp_pred=t_test[:, 0, 0, 0],
        spac_pred=r_test_c[0, :, 0, 0],
        field_pred=cs_c[:, :, 0, 0],
        time_stamps=[0, 200, 400],
        xlabel=r"r ($\mu$m)",
        ylabel=r"C$_{s,Li,ca}$ (kmol/m$^3$)",
        title=f"Step {stepID}",
        file_path_name=file_path_name,
        temp_dat=temp_dat,
        spac_dat=spac_dat,
        field_dat=field_dat,
        verbose=False,
    )
    plotData(
        [r_test_a[:, :, 0, 0], r_test_c[:, :, 0, 0]],
        [cs_a[:, :, 0, 0], cs_c[:, :, 0, 0]],
        tmax,
        [r"$[kmol/m^3]$", r"$[kmol/m^3]$"],
        [r"$c_{s,an}$", r"$c_{s,ca}$"],
        ["r [m]", "r [m]"],
        # vminList=[17, 19],
        # vmaxList=[27, 43],
    )
    plt.savefig(
        os.path.join(figureFolder, f"cs2D{string_params}_{indMov}.png")
    )

    # PHI
    file_path_name = os.path.join(
        figureFolder, f"phi{string_params}_{indMov}.png"
    )
    try:
        temp_dat = data_dict["phie_t"]
        field_phie_dat = data_dict["phie"]
        field_phis_c_dat = data_dict["phis_c"]
    except:
        temp_dat = None
        field_phie_dat = None
        field_phis_c_dat = None

    line_phi_results(
        temp_pred=t_test[:, 0, 0, 0],
        field_phie_pred=phie[:, 0, 0, 0],
        field_phis_c_pred=phis_c[:, 0, 0, 0],
        file_path_name=file_path_name,
        title=f"Step {stepID}",
        temp_dat=temp_dat,
        field_phie_dat=field_phie_dat,
        field_phis_c_dat=field_phis_c_dat,
        verbose=False,
    )


def makeCorrPlot(movieDir, dataDict, predDict, params_list, indMov, stepID):
    figureFolder = movieDir
    path = Path(figureFolder)
    path.mkdir(parents=True, exist_ok=True)
    string_params = from_param_list_to_str(params_list)

    fig, axs = plt.subplots(2, 2, figsize=(2 * 3, 2 * 3))
    axs[0, 0].plot(dataDict["phie"], predDict["phie"], "o", color="k")
    axs[0, 0].plot(dataDict["phie"], dataDict["phie"], "--", color="k")
    axprettyLabels(axs[0, 0], r"$\phi_{e,PDE}$", r"$\phi_{e,PINN}$", 14)
    axs[0, 1].plot(dataDict["phis_c"], predDict["phis_c"], "o", color="k")
    axs[0, 1].plot(dataDict["phis_c"], dataDict["phis_c"], "--", color="k")
    axprettyLabels(axs[0, 1], r"$\phi_{s,c,PDE}$", r"$\phi_{s,c,PINN}$", 14)
    axs[1, 0].plot(dataDict["cs_a"], predDict["cs_a"], "o", color="k")
    axs[1, 0].plot(dataDict["cs_a"], dataDict["cs_a"], "--", color="k")
    axprettyLabels(axs[1, 0], r"$c_{s,a,PDE}$", r"$c_{s,a,PINN}$", 14)
    axs[1, 1].plot(dataDict["cs_c"], predDict["cs_c"], "o", color="k")
    axs[1, 1].plot(dataDict["cs_c"], dataDict["cs_c"], "--", color="k")
    axprettyLabels(axs[1, 1], r"$c_{s,c,PDE}$", r"$c_{s,c,PINN}$", 14)
    currentAxis = 2
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(
        "Step " + str(stepID),
        fontsize=14,
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.savefig(
        os.path.join(
            figureFolder,
            f"corr{string_params}_{indMov}.png",
        )
    )
    plt.close()


def log(inpt, name):
    if optimized == False:
        print(name)
        print("min %.2f max %.2f " % (np.amin(inpt), np.amax(inpt)))


# Read command line arguments
args = argument.initArg()

if args.simpleModel:
    from spm_simpler import *
else:
    from spm import *

params = makeParams()

if args.params_list is None:
    params_list = None
    sys.exit("param list is mandatory here")
else:
    params_list = [float(param_val) for param_val in args.params_list]

if not args.verbose:
    import matplotlib

    matplotlib.use("Agg")

dataFolder = args.dataFolder
modelFolder = args.modelFolder
print("INFO: Using modelFolder : ", modelFolder)


if not os.path.exists(os.path.join(modelFolder, "config.json")):
    print("Looking for file ", os.path.join(modelFolder, "config.json"))
    sys.exit("ERROR: Config file could not be found necessary")
else:
    print("INFO: Loading from config file")
    with open(os.path.join(modelFolder, "config.json")) as json_file:
        configDict = json.load(json_file)
    nn = initialize_nn_from_params_config(params, configDict)

model = nn.model
sol_true = None
if dataFolder is None:
    data_dict = None
    data_dict_struct = None
    var_dict = None
    params_dict = None
else:
    try:
        print("INFO: Using dataFolder : ", dataFolder)
        data_dict = make_data_dict(dataFolder, params_list)
        data_dict_struct = make_data_dict_struct(dataFolder, params_list)
        var_dict, params_dict = make_var_params_from_data(nn, data_dict)
        sol_true = True
    except FileNotFoundError:
        data_dict = None
        data_dict_struct = None
        var_dict = None
        params_dict = None


# Get list of models
step_float_sorted, step_str_sorted = getModelStep(modelFolder)
movieDir = "Figures/Movie"
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
path = Path(os.path.join(figureFolder, modelFolderFig))
path.mkdir(parents=True, exist_ok=True)
for i_model, step in enumerate(step_str_sorted):
    nn = safe_load(nn, os.path.join(modelFolder, f"step_{step}.h5"))
    sol_dict = pinn_pred_struct(nn, params_list)
    makePlot(
        movieDir,
        sol_dict,
        params_list,
        i_model,
        int(step),
        data_dict=data_dict_struct,
    )
    if sol_true is not None:
        pred_dict = pinn_pred(nn, var_dict, params_dict)
        makeCorrPlot(
            movieDir, data_dict, pred_dict, params_list, i_model, int(step)
        )

param_string = from_param_list_to_str(params_list)

makeMovie(
    len(step_str_sorted),
    movieDir,
    os.path.join(figureFolder, modelFolderFig, f"line_cs_a{param_string}.gif"),
    prefix=f"cs_a{param_string}_",
)
makeMovie(
    len(step_str_sorted),
    movieDir,
    os.path.join(figureFolder, modelFolderFig, f"line_cs_c{param_string}.gif"),
    prefix=f"cs_c{param_string}_",
)
makeMovie(
    len(step_str_sorted),
    movieDir,
    os.path.join(figureFolder, modelFolderFig, f"cs2D{param_string}.gif"),
    prefix=f"cs2D{param_string}_",
)
makeMovie(
    len(step_str_sorted),
    movieDir,
    os.path.join(figureFolder, modelFolderFig, f"phi{param_string}.gif"),
    prefix=f"phi{param_string}_",
)
if sol_true is not None:
    makeMovie(
        len(step_str_sorted),
        movieDir,
        os.path.join(figureFolder, modelFolderFig, f"corr{param_string}.gif"),
        prefix=f"corr{param_string}_",
    )
