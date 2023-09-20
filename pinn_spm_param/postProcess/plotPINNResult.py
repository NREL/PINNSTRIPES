import json
import os
import sys

import numpy as np

sys.path.append("../util")
import argument
import tensorflow as tf
from forwardPass import (
    from_param_list_to_str,
    make_data_dict_struct,
    pinn_pred_struct,
)
from init_pinn import initialize_nn_from_params_config, safe_load
from myNN import *
from plotsUtil import *
from plotsUtil_batt import *
from tensorflow import keras
from tensorflow.keras import layers, regularizers

tf.keras.backend.set_floatx("float64")

print("\n\nINFO: PLOTTING RESULTS OF THE PINN TRAINING\n\n")


def log(inpt, name):
    if optimized == False:
        print(name)
        print("min %.2f max %.2f " % (np.amin(inpt), np.amax(inpt)))


def plot_pinn_result(args):
    if args.simpleModel:
        from spm_simpler import makeParams
    else:
        from spm import makeParams

    params = makeParams()

    input_params = False
    if not args.params_list is None:
        input_params = True
        params_list = [float(entry) for entry in args.params_list]
    else:
        sys.exit("ERROR: param list is mandatory here")

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

    nn = safe_load(nn, (os.path.join(modelFolder, "best.h5")))
    model = nn.model

    sol_dict = pinn_pred_struct(nn, params_list)

    if dataFolder is None:
        data_dict = False
    else:
        try:
            print("INFO: Using dataFolder : ", dataFolder)
            data_dict = make_data_dict_struct(dataFolder, params_list)
        except FileNotFoundError:
            data_dict = False

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

    phie = sol_dict["phie"]
    phis_c = sol_dict["phis_c"]
    cs_a = sol_dict["cs_a"]
    cs_c = sol_dict["cs_c"]
    t_test = sol_dict["t_test"]
    r_test_a = sol_dict["r_test_a"]
    r_test_c = sol_dict["r_test_c"]
    tmax = sol_dict["tmax"]

    string_params = from_param_list_to_str(params_list)
    # Plot result battery style
    # CS A
    file_path_name = os.path.join(
        figureFolder, modelFolderFig, f"line_cs_a{string_params}.png"
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
        file_path_name=file_path_name,
        temp_dat=temp_dat,
        spac_dat=spac_dat,
        field_dat=field_dat,
        verbose=args.verbose,
    )
    # CS C
    file_path_name = os.path.join(
        figureFolder, modelFolderFig, f"line_cs_c{string_params}.png"
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
        file_path_name=file_path_name,
        temp_dat=temp_dat,
        spac_dat=spac_dat,
        field_dat=field_dat,
        verbose=args.verbose,
    )
    plotData(
        [r_test_a[:, :, 0, 0], r_test_c[:, :, 0, 0]],
        [cs_a[:, :, 0, 0], cs_c[:, :, 0, 0]],
        tmax,
        [r"$[kmol/m^3]$", r"$[kmol/m^3]$"],
        [r"$c_{s,an}$", r"$c_{s,ca}$"],
        ["r [m]", "r [m]"],
    )
    if not args.verbose:
        plt.savefig(
            os.path.join(
                figureFolder,
                modelFolderFig,
                f"CSPINNResults{string_params}.png",
            )
        )

    # PHI
    file_path_name = os.path.join(
        figureFolder, modelFolderFig, f"line_phi{string_params}.png"
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
        temp_dat=temp_dat,
        field_phie_dat=field_phie_dat,
        field_phis_c_dat=field_phis_c_dat,
        verbose=args.verbose,
    )

    if configDict["dynamicAttentionWeights"]:
        if activeBound:
            t_bound_col = np.load(os.path.join(modelFolder, "t_bound_col.npy"))
            r_min_bound_col = np.load(
                os.path.join(modelFolder, "r_min_bound_col.npy")
            )
            r_maxa_bound_col = np.load(
                os.path.join(modelFolder, "r_maxa_bound_col.npy")
            )
            r_maxc_bound_col = np.load(
                os.path.join(modelFolder, "r_maxc_bound_col.npy")
            )
            bound_col_weights = np.load(
                os.path.join(modelFolder, "sa_bound_weights.npy"),
                allow_pickle=True,
            )
            dummyRbound = np.reshape(
                np.linspace(
                    params["rmin"], params["Rs_a"], len(r_maxa_bound_col)
                ),
                r_maxa_bound_col.shape,
            )
        if configDict["activeInt"]:
            t_int_col = np.load(os.path.join(modelFolder, "t_int_col.npy"))
            r_a_int_col = np.load(os.path.join(modelFolder, "r_a_int_col.npy"))
            r_c_int_col = np.load(os.path.join(modelFolder, "r_c_int_col.npy"))
            r_maxa_int_col = np.load(
                os.path.join(modelFolder, "r_maxa_int_col.npy")
            )
            r_maxc_int_col = np.load(
                os.path.join(modelFolder, "r_maxc_int_col.npy")
            )
            int_col_weights = np.load(
                os.path.join(modelFolder, "sa_int_weights.npy"),
                allow_pickle=True,
            )
        if configDict["activeData"]:
            t_data_col = np.load(
                os.path.join(modelFolder, "t_data_col.npy"), allow_pickle=True
            )
            r_data_col = np.load(
                os.path.join(modelFolder, "r_data_col.npy"), allow_pickle=True
            )
            data_col_weights = np.load(
                os.path.join(modelFolder, "sa_data_weights.npy"),
                allow_pickle=True,
            )
        if configDict["activeReg"]:
            t_reg_col = np.load(os.path.join(modelFolder, "t_reg_col.npy"))
            reg_col_weights = np.load(
                os.path.join(modelFolder, "sa_reg_weights.npy"),
                allow_pickle=True,
            )

        # collocation point weights
        if configDict["activeInt"]:
            vmax = np.amax(int_col_weights)
            vmin = np.amin(int_col_weights)
            plotCollWeights(
                [
                    r_a_int_col,
                    r_a_int_col,
                    r_a_int_col,
                    r_c_int_col,
                ],
                [
                    t_int_col,
                    t_int_col,
                    t_int_col,
                    t_int_col,
                ],
                [
                    int_col_weights[0],
                    int_col_weights[1],
                    int_col_weights[2],
                    int_col_weights[3],
                ],
                tmax,
                [
                    r"$\phi_{e}$",
                    r"$\phi_{s,ca}$",
                    r"$c_{s,an}$",
                    r"$c_{s,ca}$",
                ],
                listXAxisName=[
                    "dummy",
                    "dummy",
                    "r",
                    "r",
                ],
                vminList=[vmin],
                vmaxList=[vmax],
                globalTitle="Interior collocation",
            )

            if not args.verbose:
                plt.savefig(
                    os.path.join(figureFolder, modelFolderFig, "SAW_int")
                    + ".png"
                )
        if configDict["activeBound"]:
            vmax = np.amax(bound_col_weights)
            vmin = np.amin(bound_col_weights)
            plotCollWeights(
                [
                    dummyRbound,
                    dummyRbound,
                    dummyRbound,
                    dummyRbound,
                ],
                [
                    t_bound_col,
                    t_bound_col,
                    t_bound_col,
                    t_bound_col,
                ],
                [
                    bound_col_weights[0],
                    bound_col_weights[1],
                    bound_col_weights[2],
                    bound_col_weights[3],
                ],
                tmax,
                [
                    r"$c_{s,R=0,an}$",
                    r"$c_{s,R=0,ca}$",
                    r"$c_{s,R=Rs,an}$",
                    r"$c_{s,R=Rs,ca}$",
                ],
                listXAxisName=[
                    "dum",
                    "dum",
                    "dum",
                    "dum",
                ],
                vminList=[vmin],
                vmaxList=[vmax],
                globalTitle="Boundary collocation",
            )

            if not args.verbose:
                plt.savefig(
                    os.path.join(figureFolder, modelFolderFig, "SAW_bound")
                    + ".png"
                )
        if configDict["activeData"]:
            vmax = np.amax(data_col_weights)
            vmin = np.amin(data_col_weights)
            plotCollWeights(
                [
                    r_data_col[0],
                    r_data_col[1],
                    r_data_col[0],
                    r_data_col[1],
                ],
                [
                    t_data_col[0],
                    t_data_col[1],
                    t_data_col[2],
                    t_data_col[3],
                ],
                [
                    data_col_weights[0],
                    data_col_weights[1],
                    data_col_weights[2],
                    data_col_weights[3],
                ],
                tmax,
                [
                    r"$\phi_{e}$",
                    r"$\phi_{s,ca}$",
                    r"$c_{s,an}$",
                    r"$c_{s,ca}$",
                ],
                listXAxisName=[
                    "dummy",
                    "dummy",
                    "r",
                    "r",
                ],
                vminList=[vmin],
                vmaxList=[vmax],
                globalTitle="Data collocation",
            )

            if not args.verbose:
                plt.savefig(
                    os.path.join(figureFolder, modelFolderFig, "SAW_data")
                    + ".png"
                )
        if configDict["activeReg"]:
            sys.exit("Reg plot not ready")

    if args.verbose:
        plt.show()


if __name__ == "__main__":
    # Read command line arguments
    args = argument.initArg()
    plot_pinn_result(args)
