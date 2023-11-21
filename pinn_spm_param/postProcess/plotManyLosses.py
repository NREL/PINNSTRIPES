import os
import sys

import numpy as np

sys.path.append("../util")
import argument
from prettyPlot.plotsUtil import pretty_labels, pretty_legend

print("\n\nINFO: PLOTTING MANY LOSSES\n\n")


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


def getLossesFolder(rootFolder, prefix="LogFin"):
    # Read Time
    loss_tmp = os.listdir(rootFolder)
    # remove non floats
    for i, entry in reversed(list(enumerate(loss_tmp))):
        if not entry.startswith(prefix):
            a = loss_tmp.pop(i)
    return loss_tmp


# Read command line arguments
args = argument.initArg()

if not args.verbose:
    import matplotlib

    matplotlib.use("Agg")

rootFolder = ".."
lossFolders = getLossesFolder(rootFolder=rootFolder)
globMSELosses = []
maxSteps = []
maxEpochs = []
for lossFolder in lossFolders:
    fileGlobalLoss = os.path.join(rootFolder, lossFolder, "log.csv")
    globMSELoss = readLoss(fileGlobalLoss)
    globMSELosses.append(globMSELoss)
    maxSteps.append(globMSELoss[-1, 1])
    maxEpochs.append(globMSELoss[-1, 0])
allSteps = np.linspace(0, min(maxSteps), 100)
allEpochs = np.linspace(0, min(maxEpochs), 100)
allLossSteps = np.zeros((100, len(globMSELosses)))
allLossEpochs = np.zeros((100, len(globMSELosses)))
for i, globMSELoss in enumerate(globMSELosses):
    allLossSteps[:, i] = np.interp(
        allSteps, globMSELoss[:, 1], globMSELoss[:, 2]
    )
    allLossEpochs[:, i] = np.interp(
        allEpochs, globMSELoss[:, 0], globMSELoss[:, 2]
    )

figureFolder = "Figures"

if not args.verbose:
    os.makedirs(figureFolder, exist_ok=True)

fig = plt.figure()
for globMSELoss in globMSELosses:
    plt.plot(globMSELoss[:, 0], globMSELoss[:, 2], linewidth=3, color="k")
pretty_labels("epoch", "Global loss", 14)
ax = plt.gca()
ax.set_yscale("log")
if not args.verbose:
    plt.savefig(os.path.join(figureFolder, "manyGlobMSE_ep") + ".png")
    plt.close()

fig = plt.figure()
for globMSELoss in globMSELosses:
    plt.plot(globMSELoss[:, 1], globMSELoss[:, 2], linewidth=3, color="k")
pretty_labels("step", "Global loss", 14)
ax = plt.gca()
ax.set_yscale("log")
if not args.verbose:
    plt.savefig(os.path.join(figureFolder, "manyGlobMSE_step") + ".png")
    plt.close()

fig = plt.figure()
mean = np.mean(allLossEpochs, axis=1)
std = np.std(allLossEpochs, axis=1)
plt.fill_between(allEpochs, mean - std, mean + std, color="k", alpha=0.2)
plt.plot(allEpochs, mean, linewidth=3, color="k")
pretty_labels("epoch", "Global loss", 14)
ax = plt.gca()
ax.set_yscale("log")
if not args.verbose:
    plt.savefig(os.path.join(figureFolder, "statGlobMSE_ep") + ".png")
    plt.close()

fig = plt.figure()
mean = np.mean(allLossSteps, axis=1)
std = np.std(allLossSteps, axis=1)
plt.fill_between(allSteps, mean - std, mean + std, color="k", alpha=0.2)
plt.plot(allSteps, mean, linewidth=3, color="k")
pretty_labels("step", "Global loss", 14)
ax = plt.gca()
ax.set_yscale("log")
if not args.verbose:
    plt.savefig(os.path.join(figureFolder, "statGlobMSE_step") + ".png")
    plt.close()

if args.verbose:
    plt.show()
