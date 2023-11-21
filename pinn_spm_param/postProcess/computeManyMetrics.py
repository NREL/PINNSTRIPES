import os
import sys

import numpy as np

sys.path.append("../util")
import re
import sys

import argument
from computeError import *
from prettyPlot.plotsUtil import plt, pretty_labels, pretty_legend
from scipy import stats

print("\n\nINFO: COMPUTING MANY METRICS\n\n")


def readLoss(filename):
    lossData = np.genfromtxt(filename, delimiter=";", skip_header=1)
    return lossData


def roundSteps(steps):
    if max(steps) >= 1000:
        start, end = 0, round(max(steps), -3)
    elif max(steps) >= 100:
        start, end = 0, round(max(steps), -2)
    elif max(steps) >= 10:
        start, end = 0, round(max(steps), -1)
    else:
        start, end = 0, max(steps)
    return start, end


def getManyFolders(rootFolder, prefix="LogFin"):
    # Read Time
    fold_tmp = os.listdir(rootFolder)
    fold_num = []
    # remove non floats
    for i, entry in reversed(list(enumerate(fold_tmp))):
        if not entry.startswith(prefix):
            a = fold_tmp.pop(i)
            # print('removed ', a)
    for entry in fold_tmp:
        num = re.findall(r"\d+", entry)
        if len(num) > 1:
            print(f"WARNING: Cannot find num of folder {entry}.")
            print("Do not trust the spearman stat")
        else:
            fold_num.append(int(num[0]))

    sortedFold = [
        x for _, x in sorted(zip(fold_num, fold_tmp), key=lambda pair: pair[0])
    ]
    return sortedFold


# Read command line arguments
args = argument.initArg()
dataFolder = args.dataFolder
if dataFolder is None:
    sys.exit("data folder needs to be defined")
if args.simpleModel:
    from spm_simpler import makeParams
else:
    from spm import makeParams
params = makeParams()


if not args.verbose:
    import matplotlib

    matplotlib.use("Agg")

rootFolder = ".."
lossFolders = getManyFolders(rootFolder=rootFolder, prefix="LogFin")
modelFolders = getManyFolders(rootFolder=rootFolder, prefix="ModelFin")

endMSELosses = []
endError = []
endError_phie = []
endError_phis_c = []
endError_cs_a = []
endError_cs_c = []
maxSteps = []
maxEpochs = []
for lossFolder in lossFolders:
    fileGlobalLoss = os.path.join(rootFolder, lossFolder, "log.csv")
    globMSELoss = readLoss(fileGlobalLoss)
    endMSELosses.append(globMSELoss[-1][-1])

if args.params_list is None:
    params_list = None
else:
    params_list = [float(param_val) for param_val in args.params_list]

from forwardPass import (
    make_data_dict,
    make_var_params_from_data,
    pinn_pred,
    pinn_pred_struct,
)

data_dict = make_data_dict(dataFolder, params_list)
for modelFolder in modelFolders:
    nn, data_dict, var_dict, params_dict = init_error(
        os.path.join(rootFolder, modelFolder),
        dataFolder,
        params,
        data_dict=data_dict,
        params_list=params_list,
    )
    pred_dict = pinn_pred(nn, var_dict, params_dict)
    globalError, err_dict = computeError(data_dict, pred_dict, debug=False)
    print(f"Global error = {globalError:.3g}")
    endError.append(globalError)
    endError_phie.append(err_dict["phie"])
    endError_phis_c.append(err_dict["phis_c"])
    endError_cs_a.append(err_dict["cs_a"])
    endError_cs_c.append(err_dict["cs_c"])

print(
    f"Loss {np.mean(np.array(endMSELosses)):.2f} +\- {np.std(np.array(endMSELosses)):.2f}"
)
print(
    f"Error {np.mean(np.array(endError)):.2f} +\- {np.std(np.array(endError)):.2f}"
)
print(
    f"Error med {np.median(np.array(endError)):.2f} ptiles {np.percentile(np.array(endError), 2.5):.2f}   +\- {np.percentile(np.array(endError), 97.5):.2f}"
)
print(
    f"\tphie {np.mean(np.array(endError_phie)):.2f} +\- {np.std(np.array(endError_phie)):.2f}"
)
print(
    f"\tphis_c {np.mean(np.array(endError_phis_c)):.2f} +\- {np.std(np.array(endError_phis_c)):.2f}"
)
print(
    f"\tcs_a {np.mean(np.array(endError_cs_a)):.2f} +\- {np.std(np.array(endError_cs_a)):.2f}"
)
print(
    f"\tcs_c {np.mean(np.array(endError_cs_c)):.2f} +\- {np.std(np.array(endError_cs_c)):.2f}"
)

if len(endError) > 1:
    try:
        R = stats.spearmanr(
            np.array(endError), np.array(endMSELosses)
        ).statistic
    except:
        R = stats.spearmanr(
            np.array(endError), np.array(endMSELosses)
        ).correlation
    print(f"Correlation Loss-Error = {R:.2f}")
