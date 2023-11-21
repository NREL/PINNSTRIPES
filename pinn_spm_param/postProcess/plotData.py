import os
import sys

import numpy as np

sys.path.append("../util")
import argument
import tensorflow as tf
from myNN import *
from plotsUtil_batt import *
from tensorflow import keras
from tensorflow.keras import layers, regularizers

tf.keras.backend.set_floatx("float64")

print("\n\nINFO: PLOTTING DATA OBTAINED FROM FINITE DIFFERENCE\n\n")


def log(inpt, name):
    if optimized == False:
        print(name)
        print("min %.2f max %.2f " % (np.amin(inpt), np.amax(inpt)))


def plot_pde_data(args):
    if args.simpleModel:
        from spm_simpler import makeParams
    else:
        from spm import makeParams

    if not args.verbose:
        import matplotlib

        matplotlib.use("Agg")

    input_params = False
    if not args.params_list is None:
        input_params = True
    if input_params:
        deg_i0_a = float(args.params_list[0])
        deg_ds_c = float(args.params_list[1])
    else:
        deg_i0_a = 0.5
        deg_ds_c = 1

    params = makeParams()

    dataFolder = args.dataFolder

    figureFolder = "Figures"
    os.makedirs(figureFolder, exist_ok=True)
    figureFolder = os.path.join(
        figureFolder, f"pde_i0a_{deg_i0_a:.2g}_dsc_{deg_ds_c:.2g}"
    )

    if not args.verbose:
        os.makedirs(figureFolder, exist_ok=True)

    if args.params_list:
        sol = np.load(
            os.path.join(
                dataFolder, f"solution_{deg_i0_a:.2g}_{deg_ds_c:.2g}.npz"
            )
        )
    else:
        sol = np.load(os.path.join(dataFolder, "solution.npz"))
    t = sol["t"]
    r_a = sol["r_a"]
    r_c = sol["r_c"]
    phie = sol["phie"]
    phis_c = sol["phis_c"]
    cs_a = sol["cs_a"]
    cs_c = sol["cs_c"]

    # Plot Phi
    fig = plt.figure()
    plt.plot(t, phie, label=r"$\phi_e$")
    plt.plot(t, phis_c, label=r"$\phi_{s,c}$")
    pretty_labels("t", "[V]", 14)
    plot_legend()
    if not args.verbose:
        plt.savefig(os.path.join(figureFolder, "PhiData.png"))

    # Plot cs
    plotData(
        [r_a, r_c],
        [cs_a, cs_c],
        params["tmax"],
        [r"$[kmol/m^3]$", r"$[kmol/m^3]$"],
        [r"$c_{s,an}$", r"$c_{s,ca}$"],
        ["r [m]", "r [m]"],
    )
    if not args.verbose:
        plt.savefig(os.path.join(figureFolder, "csData.png"))

    if args.verbose:
        plt.show()


if __name__ == "__main__":
    # Read command line arguments
    args = argument.initArg()
    plot_pde_data(args)
