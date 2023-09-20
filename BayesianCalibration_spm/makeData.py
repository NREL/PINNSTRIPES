import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Make data for bayes cal")
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

sys.path.append(args.utilFolder)

import argument
import numpy as np
from forwardPass import from_param_list_to_str
from plotsUtil import *
from plotsUtil_batt import *

# Read command line arguments
args_spm = argument.initArg()

if args_spm.simpleModel:
    from spm_simpler import *
else:
    from spm import *

if not args_spm.verbose:
    import matplotlib

    matplotlib.use("Agg")

params = makeParams()


if not args_spm.params_list is None:
    input_params = True
    params_list = [float(entry) for entry in args_spm.params_list]
else:
    sys.exit("ERROR: param list is mandatory here")

deg_dict = {"i0_a": params_list[0], "ds_c": params_list[1]}
print("INFO: DEG PARAM = ", deg_dict)


dataFolder = args_spm.dataFolder


def hypercube_combinations(val_list):
    if val_list:
        for el in val_list[0]:
            for combination in hypercube_combinations(val_list[1:]):
                yield [el] + combination
    else:
        yield []


verts = [[0, 1] for _ in range(2)]
combs = hypercube_combinations(verts)

solData_list = []
params["deg_params_names"] = ["i0_a", "ds_c"]

for comb in combs:
    par_list = []
    for ipar, name in enumerate(params["deg_params_names"]):
        if comb[ipar] == 0:
            par_list.append(params["deg_" + name + "_min"])
        elif comb[ipar] == 1:
            par_list.append(params["deg_" + name + "_max"])
    param_string = from_param_list_to_str(par_list)
    solData = np.load(os.path.join(dataFolder, f"solution{param_string}.npz"))
    solData_list.append(solData)

param_string = from_param_list_to_str(params_list)
solData_meas = np.load(os.path.join(dataFolder, f"solution{param_string}.npz"))


nT_target = args.n_t
t_measure = np.linspace(params["tmin"] + 10, params["tmax"], nT_target)
data_measure = np.interp(t_measure, solData_meas["t"], solData_meas["phis_c"])
noise = args.noise
data_measure += np.random.normal(0, noise, nT_target)


import matplotlib.pyplot as plt

fig = plt.figure()
for solData in solData_list:
    plt.plot(solData["t"], solData["phis_c"], "--", color="k")

plt.plot(
    t_measure,
    data_measure,
    linewidth=3,
    color="r",
)
prettyLabels("t [s]", r"$\phi_{s,+}$ [V]", 14)

if args_spm.verbose:
    plt.show()
else:
    figureFolder = "Figures"
    os.makedirs(figureFolder, exist_ok=True)
    plt.savefig(
        os.path.join(figureFolder, f"measurement_{nT_target}_{noise:.2g}.png")
    )
    plt.close()


np.savez(
    f"dataMeasured_{nT_target}_{noise:.2g}",
    data=data_measure.astype("float64"),
    t=t_measure.astype("float64"),
    deg_params=np.array(params_list).astype("float64"),
)
