import os
import sys

import numpy as np

sys.path.append("../util")
import argument
import tensorflow as tf
from pinn import *
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


# Read command line arguments
args = argument.initArg()

from spm_simpler import *

params = makeParams()

input_params = False
if not args.params_list is None:
    input_params = True
    params_list = [float(entry) for entry in args.params_list]
else:
    sys.exit("param list is mandatory here")

if not args.verbose:
    import matplotlib

    matplotlib.use("Agg")


modelFolder = args.modelFolder
print("INFO: Using modelFolder : ", modelFolder)


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
    hidden_units_t=hidden_units_t,
    hidden_units_t_r=hidden_units_t_r,
    hidden_units_phie=hidden_units_phie,
    hidden_units_phis_c=hidden_units_phis_c,
    hidden_units_cs_a=hidden_units_cs_a,
    hidden_units_cs_c=hidden_units_cs_c,
    n_hidden_res_blocks=n_hidden_res_blocks,
    n_res_block_layers=n_res_block_layers,
    n_res_block_units=n_res_block_units,
    hard_IC_timescale=np.float64(HARD_IC_TIMESCALE),
    exponentialLimiter=EXP_LIMITER,
    activation=ACTIVATION,
    linearizeJ=LINEARIZE_J,
    params_min=params_min,
    params_max=params_max,
)

model = nn.model
model.load_weights(os.path.join(modelFolder, "best.h5"))

n_t = 100
n_r = 100
n_par = 1
tmin = 0
tmax = params["tmax"]
rmin = params["rmin"]
rmax_a = params["Rs_a"]
rmax_s = max(params["Rs_a"], params["Rs_c"])
rmax_c = params["Rs_c"]

t_test = np.reshape(np.linspace(tmin, tmax, n_t), (n_t, 1, 1, 1))
t_test_a = np.reshape(np.linspace(tmin, tmax, n_t), (n_t, 1, 1, 1))
r_test_a = np.reshape(np.linspace(rmin, rmax_a, n_r), (1, n_r, 1, 1))

t_a = np.reshape(
    np.repeat(
        np.repeat(np.repeat(t_test_a, n_r, axis=1), n_par, axis=2),
        n_par,
        axis=3,
    ),
    (n_r * n_t * n_par * n_par, 1),
).astype("float64")
r_a = np.reshape(
    np.repeat(
        np.repeat(np.repeat(r_test_a, n_t, axis=0), n_par, axis=2),
        n_par,
        axis=3,
    ),
    (n_r * n_t * n_par * n_par, 1),
).astype("float64")
r_surf_a = np.reshape(
    np.linspace(rmax_a, rmax_a, n_t * n_r * n_par * n_par),
    (n_t * n_r * n_par * n_par, 1),
).astype("float64")

t_test_c = np.reshape(np.linspace(tmin, tmax, n_t), (n_t, 1, 1, 1))
r_test_c = np.reshape(np.linspace(rmin, rmax_c, n_r), (1, n_r, 1, 1))
t_c = np.reshape(
    np.repeat(
        np.repeat(np.repeat(t_test_c, n_r, axis=1), n_par, axis=2),
        n_par,
        axis=3,
    ),
    (n_r * n_t * n_par * n_par, 1),
).astype("float64")
r_c = np.reshape(
    np.repeat(
        np.repeat(np.repeat(r_test_c, n_t, axis=0), n_par, axis=2),
        n_par,
        axis=3,
    ),
    (n_r * n_t * n_par * n_par, 1),
).astype("float64")
r_surf_c = np.reshape(
    np.linspace(rmax_c, rmax_c, n_t * n_r * n_par * n_par),
    (n_t * n_r * n_par * n_par, 1),
).astype("float64")
params_r = [
    np.ones((n_t * n_r, 1)) * (entry / nn.resc_params[ientry])
    for ientry, entry in enumerate(params_list)
]

resc_r = params["rescale_R"]
resc_t = params["rescale_T"]

out_all = nn.model([t_a / resc_t, r_a / resc_r] + params_r)
out_a = nn.model([t_a / resc_t, r_a / resc_r] + params_r)
out_surf_a = nn.model([t_a / resc_t, r_surf_a / resc_r] + params_r)
out_c = nn.model([t_c / resc_t, r_c / resc_r] + params_r)
out_surf_c = nn.model([t_c / resc_t, r_surf_c / resc_r] + params_r)

cs_surf_a = np.reshape(
    nn.rescaleCs_a(out_surf_a[nn.ind_cs_a], t_a), (n_t, n_r, n_par, n_par)
)
cs_surf_c = np.reshape(
    nn.rescaleCs_c(out_surf_c[nn.ind_cs_c], t_c), (n_t, n_r, n_par, n_par)
)
i0_a_a = params["i0_a"](
    cs_surf_a,
    params["ce0"] * np.ones(t_a.shape),
    params["T"],
    params["alpha_a"],
    params["csanmax"],
    params["R"],
    params_r[0] * nn.resc_params[0],
)
phie = np.reshape(
    nn.rescalePhie(out_all[nn.ind_phie], t_a, i0_a_a), (n_t, n_r, n_par, n_par)
)

phis_c = np.reshape(
    nn.rescalePhis_c(out_c[nn.ind_phis_c], t_c, i0_a_a),
    (n_t, n_r, n_par, n_par),
)

cs_surf_a = np.reshape(
    nn.rescaleCs_a(out_surf_a[nn.ind_cs_a], t_a), (n_t, n_r, n_par, n_par)
)

cs_a = np.reshape(
    nn.rescaleCs_a(out_a[nn.ind_cs_a], t_a), (n_t, n_r, n_par, n_par)
)
cs_c = np.reshape(
    nn.rescaleCs_c(out_c[nn.ind_cs_c], t_c), (n_t, n_r, n_par, n_par)
)

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

#         vminList=[1.13,    -0.19,      -7.5e-5,         3.55,            4,            20],
#         vmaxList=[1.265,   -0.02,            0,          4.4,           26,            45])


# Plot result battery style
ind200 = np.argmin(abs(t_test - 200))
mincs = np.amin(cs_a[:ind200]) * 0.9
maxcs = np.amax(cs_a[:ind200]) * 1.1
fig = plt.figure()
plt.plot(
    r_test_a[0, :, 0, 0] * 1e6,
    cs_a[0, :, 0, 0],
    linewidth=3,
    color="k",
    label="t=0s",
)
plt.plot(
    r_test_a[0, :, 0, 0] * 1e6,
    cs_a[ind200, :, 0, 0],
    linewidth=3,
    color="0.3",
    label="t=200s",
)
prettyLabels(r"r ($\mu$m)", r"C$_{s,Li,an}$ (kmol/m$^3$)", 14)
plotLegend()
if not args.verbose:
    string_params = ["_%.2g" for _ in params_list]
    plt.savefig(
        os.path.join(
            figureFolder,
            modelFolderFig,
            "line_cs_a" + "".join(string_params) % tuple(params_list),
        )
        + ".png"
    )
    plt.close()
# Plot result battery style
ind200 = np.argmin(abs(t_test - 200))
mincs = np.amin(cs_c[:ind200]) * 0.9
maxcs = np.amax(cs_c[:ind200]) * 1.1
fig = plt.figure()
plt.plot(
    r_test_c[0, :, 0, 0] * 1e6,
    cs_c[0, :, 0, 0],
    linewidth=3,
    color="k",
    label="t=0s",
)
plt.plot(
    r_test_c[0, :, 0, 0] * 1e6,
    cs_c[ind200, :, 0, 0],
    linewidth=3,
    color="0.3",
    label="t=200s",
)
prettyLabels(r"r ($\mu$m)", r"C$_{s,Li,ca}$ (kmol/m$^3$)", 14)
plotLegend()
if not args.verbose:
    string_params = ["_%.2g" for _ in params_list]
    plt.savefig(
        os.path.join(
            figureFolder,
            modelFolderFig,
            "line_cs_c" + "".join(string_params) % tuple(params_list),
        )
        + ".png"
    )
    plt.close()

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
# vminList=[ 4,            20],
# vmaxList=[26,            45])
if not args.verbose:
    string_params = ["_%.2g" for _ in params_list]
    plt.savefig(
        os.path.join(
            figureFolder,
            modelFolderFig,
            "CSPINNResults" + "".join(string_params) % tuple(params_list),
        )
        + ".png"
    )

fig = plt.figure()
plt.plot(
    t_test[:, :, 0, 0],
    phis_c[:, 0, 0, 0],
    linewidth=3,
    color="k",
    label=r"$\phi_{s,c}$",
)
plt.plot(
    t_test[:, :, 0, 0],
    phie[:, 0, 0, 0],
    linewidth=3,
    color="b",
    label=r"$\phi_{e}$",
)
prettyLabels("t", "[V]", 14)
plotLegend()

if not args.verbose:
    string_params = ["_%.2g" for _ in params_list]
    plt.savefig(
        os.path.join(
            figureFolder,
            modelFolderFig,
            "PhiPINNResults" + "".join(string_params) % tuple(params_list),
        )
        + ".png"
    )


if DYNAMIC_ATTENTION:
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
            np.linspace(params["rmin"], params["Rs_a"], len(r_maxa_bound_col)),
            r_maxa_bound_col.shape,
        )
    if activeInt:
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
            os.path.join(modelFolder, "sa_int_weights.npy"), allow_pickle=True
        )
    if activeData:
        t_data_col = np.load(
            os.path.join(modelFolder, "t_data_col.npy"), allow_pickle=True
        )
        r_data_col = np.load(
            os.path.join(modelFolder, "r_data_col.npy"), allow_pickle=True
        )
        data_col_weights = np.load(
            os.path.join(modelFolder, "sa_data_weights.npy"), allow_pickle=True
        )
    if activeReg:
        t_reg_col = np.load(os.path.join(modelFolder, "t_reg_col.npy"))
        reg_col_weights = np.load(
            os.path.join(modelFolder, "sa_reg_weights.npy"), allow_pickle=True
        )

    # collocation point weights
    if activeInt:
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
                os.path.join(figureFolder, modelFolderFig, "SAW_int") + ".png"
            )
    if activeBound:
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
    if activeData:
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
                os.path.join(figureFolder, modelFolderFig, "SAW_data") + ".png"
            )
    if activeReg:
        sys.exit("Reg plot not ready")

if args.verbose:
    plt.show()
