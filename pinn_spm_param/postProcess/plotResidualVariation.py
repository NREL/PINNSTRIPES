import os
import sys

import numpy as np

sys.path.append("../util")
import argument
from plotsUtil import *

print("\n\nINFO: PLOTTING LOSSES\n\n")


def fromStr2Arr(string):
    string = string[1:-2]
    stringArr = string.split(",")
    stringArr = [float(entry) for entry in stringArr]
    return stringArr


def readLoss(filename):
    lossData = np.genfromtxt(filename, delimiter=";", skip_header=1)
    return lossData


def readResiduals(filename):
    f = open(filename, "r+")
    lines = f.readlines()
    # Skip first line
    lines = lines[1:]
    # Get format
    line = lines[0]
    line_component = line.split(";")
    tmp_arr = fromStr2Arr(line_component[1])
    nEntries = len(tmp_arr)
    nLines = len(lines)
    # Allocate
    steps = np.zeros(nLines // 2)
    residuals = np.zeros((nLines // 2, nEntries))
    residualsProportion = np.zeros((nLines // 2, nEntries))
    # Read
    countResidual = 0
    for i in range(nLines):
        line_component = lines[i].split(";")
        arr = fromStr2Arr(line_component[1])
        if i % 2 == 1:
            steps[countResidual] = int(line_component[0])
            for j in range(nEntries):
                residuals[countResidual, j] = arr[j]
                residualsProportion[countResidual, j] = arr[j] / sum(arr)
            countResidual += 1
    f.close()
    return steps, residuals, residualsProportion


def plotProportion(
    steps, residuals, residualsProportion, names, title, ax=None
):
    order = list(np.argsort(residuals[0, :])[::-1])
    if ax == None:
        fig = plt.figure()
    curvelist = []
    for iind, ind in enumerate(order):
        up = np.zeros(residuals.shape[0])
        for i in range(iind + 1):
            up += residualsProportion[:, order[i]]
        curvelist.append(up)
    for icurve, curve in enumerate(curvelist):
        if icurve == 0:
            y1 = curvelist[icurve]
            y2 = 0
        else:
            y1 = curvelist[icurve]
            y2 = curvelist[icurve - 1]
        maxVal = np.amax(y1 - y2)
        if maxVal > 0.1:
            if ax == None:
                plt.fill_between(steps, y1, y2, label=names[order[icurve]])
            else:
                ax.fill_between(steps, y1, y2, label=names[order[icurve]])
        else:
            if ax == None:
                plt.fill_between(steps, y1, y2)
            else:
                ax.fill_between(steps, y1, y2)
    if ax == None:
        prettyLabels("step", "loss proportion", 14, title=title)
        plotLegend()
    else:
        axprettyLabels(ax, "step", "loss proportion", 14, title=title)
        axplotLegend(ax)


def plotConvergence(steps, residuals, names, title, ax=None):
    if ax == None:
        fig = plt.figure()
    order = list(np.argsort(residuals[-1, :] / residuals[0, :])[::-1])
    for ind in order:
        if ax == None:
            plt.plot(steps, residuals[:, ind], label=names[ind])
        else:
            ax.plot(steps, residuals[:, ind], label=names[ind])
    if ax == None:
        axis = plt.gca()
        axis.set_yscale("log")
    else:
        ax.set_yscale("log")
    if ax == None:
        prettyLabels("step", "residual", 14)
        plotLegend()
    else:
        axprettyLabels(ax, "step", "residual", 14)
        axplotLegend(ax)


def plotConvergence_rescaledInit(steps, residuals, names, title, ax=None):
    if ax == None:
        fig = plt.figure()
    order = list(np.argsort(residuals[-1, :] / residuals[0, :])[::-1])
    for ind in order:
        if ax == None:
            plt.plot(
                steps, residuals[:, ind] / residuals[0, ind], label=names[ind]
            )
        else:
            ax.plot(
                steps, residuals[:, ind] / residuals[0, ind], label=names[ind]
            )
    if ax == None:
        axis = plt.gca()
        axis.set_yscale("log")
        prettyLabels("step", "residual rescaled", 14, title=title)
        plotLegend()
    else:
        ax.set_yscale("log")
        axprettyLabels(ax, "step", "residual rescaled", 14, title=title)
        axplotLegend(ax)


def plotConvergence_rescaledInitFilt(
    steps, residuals, names, title, filterSize=10, ax=None
):
    filterSize = min(filterSize, residuals.shape[0] // 2)
    kernel = [1.0 / filterSize for _ in range(filterSize)]
    filtSteps = np.convolve(steps, kernel, mode="valid")
    filtRes = np.zeros(
        (residuals.shape[0] - filterSize + 1, residuals.shape[1])
    )
    for i in range(filtRes.shape[1]):
        filtRes[:, i] = np.convolve(residuals[:, i], kernel, mode="valid")
    steps = filtSteps
    residuals = filtRes
    if ax == None:
        fig = plt.figure()
    order = list(np.argsort(residuals[-1, :] / residuals[0, :])[::-1])
    for ind in order:
        if ax == None:
            plt.plot(
                steps, residuals[:, ind] / residuals[0, ind], label=names[ind]
            )
        else:
            ax.plot(
                steps, residuals[:, ind] / residuals[0, ind], label=names[ind]
            )
    if ax == None:
        axis = plt.gca()
        axis.set_yscale("log")
        prettyLabels("step", "residual rescaled", 14, title=title)
        plotLegend()
    else:
        ax.set_yscale("log")
        axprettyLabels(ax, "step", "residual rescaled", 14, title=title)
        axplotLegend(ax)


# Read command line arguments
args = argument.initArg()

if not args.verbose:
    import matplotlib

    matplotlib.use("Agg")

logFolder = args.logFolder
modelFolder = args.modelFolder
print("INFO: Using logFolder : ", logFolder)
print("INFO: Using modelFolder : ", modelFolder)

if not os.path.exists(os.path.join(modelFolder, "config.npy")):
    activeInt = False
    activeBound = False
    activeData = False
    activeReg = False
    DYNAMIC_ATTENTION = False
else:
    print("INFO: Loading from config file")
    configDict = np.load(
        os.path.join(modelFolder, "config.npy"), allow_pickle="TRUE"
    ).item()
    activeInt = configDict["activeInt"]
    activeBound = configDict["activeBound"]
    activeData = configDict["activeData"]
    activeReg = configDict["activeReg"]
    DYNAMIC_ATTENTION = configDict["dynamicAttentionWeights"]


fileBoundaryRes = os.path.join(logFolder, "boundaryTerms.csv")
fileInteriorRes = os.path.join(logFolder, "interiorTerms.csv")
fileDataRes = os.path.join(logFolder, "dataTerms.csv")
fileRegRes = os.path.join(logFolder, "regTerms.csv")
fileGlobalLoss = os.path.join(logFolder, "log.csv")


ResBoundName = [
    r"C$_{s,r=0,an}$",
    r"C$_{s,r=0,cath}$",
    r"C$_{s,r=Rs,an}$",
    r"C$_{s,r=Rs,cath}$",
]
ResIntName = [
    r"$\phi_{e}$",
    r"$\phi_{s,cath}$",
    r"C$_{s,an}$",
    r"C$_{s,cath}$",
]
ResDataName = [r"$\phi_{e}$", r"$\phi_{s,cath}$", r"C$_{s,an}$", r"C$_{s,ca}$"]

ResRegName = []

passBoundaryCheck = True
try:
    stepsBound, residualsBound, residualsBoundProportion = readResiduals(
        fileBoundaryRes
    )
except:
    passBoundaryCheck = False
if activeBound:
    if not len(ResBoundName) == residualsBound.shape[1]:
        print(
            "Mismatch in boundary names and residuals. Update boundary residual names"
        )
        passBoundaryCheck = False

passInteriorCheck = True
try:
    stepsInt, residualsInt, residualsIntProportion = readResiduals(
        fileInteriorRes
    )
except:
    passInteriorCheck = False
if activeInt:
    if not len(ResIntName) == residualsInt.shape[1]:
        print(
            "Mismatch in physics names and residuals. Update physics residual names"
        )
        passInteriorCheck = False

passDataCheck = True
try:
    stepsData, residualsData, residualsDataProportion = readResiduals(
        fileDataRes
    )
except:
    passDataCheck = False
if activeData:
    if not len(ResDataName) == residualsData.shape[1]:
        print(
            "Mismatch in data names and residuals. Update data residual names"
        )
        passDataCheck = False

passRegCheck = True
try:
    stepsReg, residualsReg, residualsRegProportion = readResiduals(fileRegRes)
except:
    passRegCheck = False
if activeReg:
    if not len(ResRegName) == residualsReg.shape[1]:
        print("Mismatch in reg names and residuals. Update Reg residual names")
        passRegCheck = False

globMSELoss = readLoss(fileGlobalLoss)

figureFolder = "Figures"
logFolderFig = (
    logFolder.replace("../", "")
    .replace("/Log", "")
    .replace("/Model", "")
    .replace("Log", "")
    .replace("Model", "")
    .replace("/", "_")
    .replace(".", "")
)

if not args.verbose:
    os.makedirs(figureFolder, exist_ok=True)
    os.makedirs(os.path.join(figureFolder, logFolderFig), exist_ok=True)


fig, axs = plt.subplots(2, 2, figsize=(2 * 5, 2 * 5))
if passBoundaryCheck:
    plotConvergence_rescaledInit(
        stepsBound,
        residualsBound,
        ResBoundName,
        title="Boundary Losses",
        ax=axs[0, 0],
    )
if passInteriorCheck:
    plotConvergence_rescaledInit(
        stepsInt,
        residualsInt,
        ResIntName,
        title="Interior Losses",
        ax=axs[0, 1],
    )
if passDataCheck:
    plotConvergence_rescaledInit(
        stepsData,
        residualsData,
        ResDataName,
        title="Data Losses",
        ax=axs[1, 0],
    )
if passRegCheck:
    plotConvergence_rescaledInit(
        stepsReg, residualsReg, ResRegName, title="Reg Losses", ax=axs[1, 1]
    )
if not args.verbose:
    try:
        plt.savefig(
            os.path.join(figureFolder, logFolderFig, "residualsVal") + ".png"
        )
    except:
        print("Could not save residual value figure")

fig, axs = plt.subplots(2, 2, figsize=(2 * 5, 2 * 5))
filterSize = 200
if passBoundaryCheck:
    plotConvergence_rescaledInitFilt(
        stepsBound,
        residualsBound,
        ResBoundName,
        title="Boundary Losses",
        filterSize=filterSize,
        ax=axs[0, 0],
    )
if passInteriorCheck:
    plotConvergence_rescaledInitFilt(
        stepsInt,
        residualsInt,
        ResIntName,
        title="Interior Losses",
        filterSize=filterSize,
        ax=axs[0, 1],
    )
if passDataCheck:
    plotConvergence_rescaledInitFilt(
        stepsData,
        residualsData,
        ResDataName,
        title="Data Losses",
        filterSize=filterSize,
        ax=axs[1, 0],
    )
if passRegCheck:
    plotConvergence_rescaledInitFilt(
        stepsReg,
        residualsReg,
        ResRegName,
        title="Reg Losses",
        filterSize=filterSize,
        ax=axs[1, 1],
    )
if not args.verbose:
    try:
        plt.savefig(
            os.path.join(figureFolder, logFolderFig, "residualsVal")
            + "_filt.png"
        )
    except:
        print("Could not save residual value filtered figure")


fig, axs = plt.subplots(2, 2, figsize=(2 * 5, 2 * 5))
if passBoundaryCheck:
    plotProportion(
        stepsBound,
        residualsBound,
        residualsBoundProportion,
        ResBoundName,
        title="Boundary Losses",
        ax=axs[0, 0],
    )
if passInteriorCheck:
    plotProportion(
        stepsInt,
        residualsInt,
        residualsIntProportion,
        ResIntName,
        title="Interior Losses",
        ax=axs[0, 1],
    )
if passDataCheck:
    plotProportion(
        stepsData,
        residualsData,
        residualsDataProportion,
        ResDataName,
        title="Data losses",
        ax=axs[1, 0],
    )
if passRegCheck:
    plotProportion(
        stepsReg,
        residualsReg,
        residualsRegProportion,
        ResRegName,
        title="Reg losses",
        ax=axs[1, 1],
    )
if not args.verbose:
    try:
        plt.savefig(
            os.path.join(figureFolder, logFolderFig, "residualsProp") + ".png"
        )
    except:
        print("Could not save residual proportion figure")

# plotProportion(stepsBound,residualsBound,residualsBoundProportion,ResBoundName,title='Boundary Losses')
# plt.savefig(os.path.join(figureFolder,'resBound_prop_') + logFolderFig + '.png')
# plotConvergence_rescaledInit(stepsBound,residualsBound,ResBoundName,title='Boundary Losses')
# plt.savefig(os.path.join(figureFolder,'resBound_rescInit_') + logFolderFig + '.png')
# plotProportion(stepsInt,residualsInt,residualsIntProportion,ResIntName,title='Interior Losses')
# plt.savefig(os.path.join(figureFolder,'resInt_prop_') + logFolderFig + '.png')
# plotConvergence_rescaledInit(stepsInt,residualsInt,ResIntName,title='Interior Losses')
# plt.savefig(os.path.join(figureFolder,'resInt_rescInit_') + logFolderFig + '.png')
# plotProportion(stepsData,residualsData,residualsDataProportion,ResDataName,title='Data losses')
# plt.savefig(os.path.join(figureFolder,'resData_prop_') + logFolderFig + '.png')
# plotConvergence_rescaledInit(stepsData,residualsData,ResDataName,title='Data Losses')
# plt.savefig(os.path.join(figureFolder,'resData_rescInit_') + logFolderFig + '.png')


fig = plt.figure()
if DYNAMIC_ATTENTION:
    # plt.plot(globMSELoss[:,0],globMSELoss[:,1],linewidth=3, color='k')
    plt.plot(globMSELoss[:, 1], linewidth=3, color="b", label="loss weighted")
    plt.plot(
        globMSELoss[:, 2], linewidth=3, color="k", label="loss unweighted"
    )
else:
    plt.plot(globMSELoss[:, 1], linewidth=3, color="k")

prettyLabels("epoch", "Global loss", 14)
if DYNAMIC_ATTENTION:
    plotLegend()
ax = plt.gca()
ax.set_yscale("log")
if not args.verbose:
    try:
        plt.savefig(
            os.path.join(figureFolder, logFolderFig, "globMSE") + ".png"
        )
    except:
        print("Could not save global loss figure")


if args.verbose:
    plt.show()