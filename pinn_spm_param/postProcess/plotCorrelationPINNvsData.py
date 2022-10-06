import os
import sys

import numpy as np

sys.path.append("../util")
import argument
import tensorflow as tf
from pinn import *
from plotsUtil import *
from tensorflow import keras
from tensorflow.keras import layers, regularizers

tf.keras.backend.set_floatx("float64")

print("\n\nINFO: PLOT CORRELATION PINN DATA\n\n")


def log(inpt, name):
    if optimized == False:
        print(name)
        print("min %.2f max %.2f " % (np.amin(inpt), np.amax(inpt)))


# Read command line arguments
args = argument.initArg()

from spm_simpler import *

params = makeParams()

input_params = False
if not args.params_list is None:
    input_params = True
if input_params:
    input_deg_i0_a = float(args.params_list[0])
    input_deg_ds_c = float(args.params_list[1])
else:
    input_deg_i0_a = 1.0
    input_deg_ds_c = 1.0

if not args.verbose:
    import matplotlib

    matplotlib.use("Agg")

modelFolder = args.modelFolder
print("INFO: Using modelFolder : ", modelFolder)
dataFolder = args.dataFolder

if not os.path.exists(os.path.join(modelFolder, "config.npy")):
    print("Looking for file ", os.path.join(modelFolder, "config.npy"))
    sys.exit("ERROR: Config file could not be found necessary")
else:
    print("INFO: Loading from config file")
    configDict = np.load(
        os.path.join(modelFolder, "config.npy"), allow_pickle="TRUE"
    ).item()
    hidden_units_t = configDict["hidden_units_t"]
    hidden_units_t_r = configDict["hidden_units_t_r"]
    hidden_units_phie = configDict["hidden_units_phie"]
    hidden_units_phis_c = configDict["hidden_units_phis_c"]
    hidden_units_cs_a = configDict["hidden_units_cs_a"]
    hidden_units_cs_c = configDict["hidden_units_cs_c"]
    try:
        n_hidden_res_blocks = configDict["n_hidden_res_blocks"]
    except:
        n_hidden_res_blocks = 0
    if n_hidden_res_blocks > 0:
        n_res_block_layers = configDict["n_res_block_layers"]
        n_res_block_units = configDict["n_res_block_units"]
    else:
        n_res_block_layers = 1
        n_res_block_units = 1
    HARD_IC_TIMESCALE = configDict["hard_IC_timescale"]
    EXP_LIMITER = configDict["exponentialLimiter"]
    ACTIVATION = configDict["activation"]
    try:
        LINEARIZE_J = configDict["linearizeJ"]
    except:
        LINEARIZE_J = True
    DYNAMIC_ATTENTION = configDict["dynamicAttentionWeights"]
    activeInt = configDict["activeInt"]
    activeBound = configDict["activeBound"]
    activeData = configDict["activeData"]
    activeReg = configDict["activeReg"]
    try:
        params_min = configDict["params_min"]
    except:
        params_min = [
            params["deg_i0_a_min"],
            params["deg_ds_c_min"],
        ]
    try:
        params_max = configDict["params_max"]
    except:
        params_max = [
            params["deg_i0_a_max"],
            params["deg_ds_c_max"],
        ]

nn = pinn(
    params=params,
    hidden_units_phie=hidden_units_phie,
    hidden_units_phis_c=hidden_units_phis_c,
    hidden_units_cs_a=hidden_units_cs_a,
    hidden_units_cs_c=hidden_units_cs_c,
    hidden_units_t=hidden_units_t,
    hidden_units_t_r=hidden_units_t_r,
    n_hidden_res_blocks=n_hidden_res_blocks,
    n_res_block_layers=n_res_block_layers,
    n_res_block_units=n_res_block_units,
    hard_IC_timescale=HARD_IC_TIMESCALE,
    exponentialLimiter=EXP_LIMITER,
    activation=ACTIVATION,
    linearizeJ=LINEARIZE_J,
    params_min=params_min,
    params_max=params_max,
)


model = nn.model
model.load_weights(os.path.join(modelFolder, "best.h5"))

if input_params:
    data_phie = np.load(
        os.path.join(
            dataFolder,
            "data_phie_%.2g_%.2g.npz" % (input_deg_i0_a, input_deg_ds_c),
        )
    )
else:
    data_phie = np.load(os.path.join(dataFolder, "data_phie.npz"))
xTest_phie = data_phie["x_test"].astype("float64")
yTest_phie = data_phie["y_test"].astype("float64")
x_params_test_phie = data_phie["x_params_test"].astype("float64")
if input_params:
    data_phis_c = np.load(
        os.path.join(
            dataFolder,
            "data_phis_c_%.2g_%.2g.npz" % (input_deg_i0_a, input_deg_ds_c),
        )
    )
else:
    data_phis_c = np.load(os.path.join(dataFolder, "data_phis_c.npz"))
xTest_phis_c = data_phis_c["x_test"].astype("float64")
yTest_phis_c = data_phis_c["y_test"].astype("float64")
x_params_test_phis_c = data_phis_c["x_params_test"].astype("float64")
if input_params:
    data_cs_a = np.load(
        os.path.join(
            dataFolder,
            "data_cs_a_%.2g_%.2g.npz" % (input_deg_i0_a, input_deg_ds_c),
        )
    )
else:
    data_cs_a = np.load(os.path.join(dataFolder, "data_cs_a.npz"))
xTest_cs_a = data_cs_a["x_test"].astype("float64")
yTest_cs_a = data_cs_a["y_test"].astype("float64")
x_params_test_cs_a = data_cs_a["x_params_test"].astype("float64")
if input_params:
    data_cs_c = np.load(
        os.path.join(
            dataFolder,
            "data_cs_c_%.2g_%.2g.npz" % (input_deg_i0_a, input_deg_ds_c),
        )
    )
else:
    data_cs_c = np.load(os.path.join(dataFolder, "data_cs_c.npz"))
xTest_cs_c = data_cs_c["x_test"].astype("float64")
yTest_cs_c = data_cs_c["y_test"].astype("float64")
x_params_test_cs_c = data_cs_c["x_params_test"].astype("float64")

# rescale
resc_r = params["rescale_R"]
resc_t = params["rescale_T"]

dummyR = np.zeros(xTest_phie[:, 0].shape)
surfR_a = params["Rs_a"] * np.ones(xTest_phie[:, 0].shape)

out = nn.model(
    [
        xTest_phie[:, 0] / resc_t,
        surfR_a / resc_r,
        x_params_test_phie[:, nn.ind_deg_i0_a]
        / nn.resc_params[nn.ind_deg_i0_a],
        x_params_test_phie[:, nn.ind_deg_ds_c]
        / nn.resc_params[nn.ind_deg_ds_c],
    ]
)
cse_a = nn.rescaleCs_a(out[nn.ind_cs_a], xTest_phie[:, 0])
ce = params["ce0"] * np.ones(cse_a.shape)


i0_a_phie = params["i0_a"](
    cse_a,
    ce,
    params["T"],
    params["alpha_a"],
    params["csanmax"],
    params["R"],
    x_params_test_phie[:, nn.ind_deg_i0_a],
)

phie_nonrescaled = out[nn.ind_phie]
phie_rescaled = nn.rescalePhie(phie_nonrescaled, xTest_phie[:, 0], i0_a_phie)

dummyR = np.zeros(xTest_phis_c[:, 0].shape)
out = nn.model(
    [
        xTest_phis_c[:, 0] / resc_t,
        dummyR,
        x_params_test_phis_c[:, nn.ind_deg_i0_a]
        / nn.resc_params[nn.ind_deg_i0_a],
        x_params_test_phis_c[:, nn.ind_deg_ds_c]
        / nn.resc_params[nn.ind_deg_ds_c],
    ]
)
phis_c_nonrescaled = out[nn.ind_phis_c]
phis_c_rescaled = nn.rescalePhis_c(
    phis_c_nonrescaled,
    xTest_phis_c[:, 0],
    i0_a_phie,
)


out = nn.model(
    [
        xTest_cs_a[:, 0] / resc_t,
        xTest_cs_a[:, 1] / resc_r,
        x_params_test_cs_a[:, nn.ind_deg_i0_a]
        / nn.resc_params[nn.ind_deg_i0_a],
        x_params_test_cs_a[:, nn.ind_deg_ds_c]
        / nn.resc_params[nn.ind_deg_ds_c],
    ]
)
cs_a_nonrescaled = out[nn.ind_cs_a]
cs_a_rescaled = nn.rescaleCs_a(cs_a_nonrescaled, xTest_cs_a[:, 0])

out = nn.model(
    [
        xTest_cs_c[:, 0] / resc_t,
        xTest_cs_c[:, 1] / resc_r,
        x_params_test_cs_c[:, nn.ind_deg_i0_a]
        / nn.resc_params[nn.ind_deg_i0_a],
        x_params_test_cs_c[:, nn.ind_deg_ds_c]
        / nn.resc_params[nn.ind_deg_ds_c],
    ]
)
cs_c_nonrescaled = out[nn.ind_cs_c]
cs_c_rescaled = nn.rescaleCs_c(cs_c_nonrescaled, xTest_cs_c[:, 0])

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

# print(yTest_cs_c-cs_c_rescaled)
# print(np.mean(abs(yTest_cs_c-cs_c_rescaled)))
# stop

fig, axs = plt.subplots(2, 2, figsize=(2 * 3, 2 * 3))
axs[0, 0].plot(yTest_phie, phie_rescaled, "o", color="k")
axs[0, 0].plot(yTest_phie, yTest_phie, "--", color="k")
axprettyLabels(axs[0, 0], r"$\phi_{e,PDE}$", r"$\phi_{e,PINN}$", 14)
axs[0, 1].plot(yTest_phis_c, phis_c_rescaled, "o", color="k")
axs[0, 1].plot(yTest_phis_c, yTest_phis_c, "--", color="k")
axprettyLabels(axs[0, 1], r"$\phi_{s,c,PDE}$", r"$\phi_{s,c,PINN}$", 14)
axs[1, 0].plot(yTest_cs_a, cs_a_rescaled, "o", color="k")
axs[1, 0].plot(yTest_cs_a, yTest_cs_a, "--", color="k")
axprettyLabels(axs[1, 0], r"$c_{s,a,PDE}$", r"$c_{s,a,PINN}$", 14)
axs[1, 1].plot(yTest_cs_c, cs_c_rescaled, "o", color="k")
axs[1, 1].plot(yTest_cs_c, yTest_cs_c, "--", color="k")
axprettyLabels(axs[1, 1], r"$c_{s,c,PDE}$", r"$c_{s,c,PINN}$", 14)
currentAxis = 2


if not args.verbose:
    if input_params:
        plt.savefig(
            os.path.join(
                figureFolder,
                modelFolderFig,
                "corrDataPINN_%.2g_%.2g" % (input_deg_i0_a, input_deg_ds_c),
            )
            + ".png"
        )
    else:
        plt.savefig(
            os.path.join(figureFolder, modelFolderFig, "corrDataPINN") + ".png"
        )


if args.verbose:
    plt.show()
