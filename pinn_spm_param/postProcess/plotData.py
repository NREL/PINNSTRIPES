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

print("\n\nINFO: PLOTTING DATA OBTAINED FROM FINITE DIFFERENCE\n\n")


def log(inpt, name):
    if optimized == False:
        print(name)
        print("min %.2f max %.2f " % (np.amin(inpt), np.amax(inpt)))


# Read command line arguments
args = argument.initArg()

from spm_simpler import *

if not args.verbose:
    import matplotlib

    matplotlib.use("Agg")

params = makeParams()

dataFolder = args.dataFolder
modelFolder = args.modelFolder

print("INFO: Using modelFolder : ", modelFolder)

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
prettyLabels("t", "[V]", 14)
plotLegend()
if not args.verbose:
    plt.savefig(os.path.join(figureFolder, modelFolderFig, "PhiData") + ".png")

# Plot cs
plotData(
    [r_a, r_c],
    [cs_a, cs_c],
    params["tmax"],
    [r"$[kmol/m^3]$", r"$[kmol/m^3]$"],
    [r"$c_{s,an}$", r"$c_{s,ca}$"],
    ["r [m]", "r [m]"],
    # vminList=[17, 19],
    # vmaxList=[27, 43],
)
if not args.verbose:
    plt.savefig(os.path.join(figureFolder, modelFolderFig, "csData") + ".png")

if args.verbose:
    plt.show()
