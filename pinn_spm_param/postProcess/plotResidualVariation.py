import json
import os
import sys

import numpy as np

sys.path.append("../util")
import argument
from prettyPlot.plotsUtil import pretty_labels, pretty_legend

print("\n\nINFO: PLOTTING LOSSES\n\n")


def fromStr2Arr(string):
    string = string[1:-2]
    stringArr = string.split(",")
    stringArr = [float(entry) for entry in stringArr]
    return stringArr


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


def read_loss_weights(filename):
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
    steps = np.zeros(nLines)
    loss_weights = np.zeros((nLines, nEntries))
    # Read
    countResidual = 0
    for i in range(nLines):
        line_component = lines[i].split(";")
        arr = fromStr2Arr(line_component[1])
        steps[countResidual] = int(line_component[0])
        for j in range(nEntries):
            loss_weights[countResidual, j] = arr[j]
        countResidual += 1
    f.close()
    return steps, loss_weights


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
        pretty_abels("step", "loss proportion", 14, title=title)
        pretty_legend()
        ax = plt.gca()
        start, end = roundSteps(steps)
        ax.xaxis.set_ticks(np.linspace(start, end, 3))
    else:
        pretty_labels(ax, "step", "loss proportion", 14, title=title)
        pretty_legend(ax=ax)
        start, end = roundSteps(steps)
        ax.xaxis.set_ticks(np.linspace(start, end, 3))


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
        pretty_labels("step", "residual", 14)
        pretty_legend()
        start, end = roundSteps(steps)
        axis.xaxis.set_ticks(np.linspace(start, end, 3))
    else:
        pretty_Labels("step", "residual", 14, ax=ax)
        pretty_legend(ax=ax)
        start, end = roundSteps(steps)
        ax.xaxis.set_ticks(np.linspace(start, end, 3))


def plotLossWeights(steps, lossWeights, names, title, ax=None):
    if ax == None:
        fig = plt.figure()
    order = list(np.argsort(lossWeights[-1, :] / lossWeights[0, :])[::-1])
    for ind in order:
        if ax == None:
            plt.plot(steps, lossWeights[:, ind], label=names[ind])
        else:
            ax.plot(steps, lossWeights[:, ind], label=names[ind])
    if ax == None:
        axis = plt.gca()
        axis.set_yscale("log")
    else:
        ax.set_yscale("log")
    if ax == None:
        pretty_labels("step", "weights", 14)
        pretty_legend()
        start, end = roundSteps(steps)
        axis.xaxis.set_ticks(np.linspace(start, end, 3))
    else:
        pretty_labels("step", "weights", 14, ax=ax)
        pretty_legend(ax=ax)
        start, end = roundSteps(steps)
        ax.xaxis.set_ticks(np.linspace(start, end, 3))


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
        pretty_labels("step", "residual rescaled", 14, title=title)
        pretty_legend()
        start, end = roundSteps(steps)
        axis.xaxis.set_ticks(np.linspace(start, end, 3))
    else:
        ax.set_yscale("log")
        pretty_labels("step", "residual rescaled", 14, title=title, ax=ax)
        pretty_legend(ax=ax)
        start, end = roundSteps(steps)
        ax.xaxis.set_ticks(np.linspace(start, end, 3))


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
        pretty_labels("step", "residual rescaled", 14, title=title)
        pretty_legend()
        start, end = roundSteps(steps)
        axis.xaxis.set_ticks(np.linspace(start, end, 3))
    else:
        ax.set_yscale("log")
        pretty_labels("step", "residual rescaled", 14, title=title, ax=ax)
        pretty_legend(ax=ax)
        start, end = roundSteps(steps)
        ax.xaxis.set_ticks(np.linspace(start, end, 3))


def plot_res_var(args):
    if not args.verbose:
        import matplotlib

        matplotlib.use("Agg")

    logFolder = args.logFolder
    modelFolder = args.modelFolder
    print("INFO: Using logFolder : ", logFolder)
    print("INFO: Using modelFolder : ", modelFolder)

    if not os.path.exists(os.path.join(modelFolder, "config.json")):
        activeInt = False
        activeBound = False
        activeData = False
        activeReg = False
        DYNAMIC_ATTENTION = False
        ANNEALING = False
    else:
        print("INFO: Loading from config file")
        with open(os.path.join(modelFolder, "config.json")) as json_file:
            configDict = json.load(json_file)
        activeInt = configDict["activeInt"]
        activeBound = configDict["activeBound"]
        activeData = configDict["activeData"]
        activeReg = configDict["activeReg"]
        DYNAMIC_ATTENTION = configDict["dynamicAttentionWeights"]
        ANNEALING = configDict["annealingWeights"]

    fileBoundaryRes = os.path.join(logFolder, "boundaryTerms.csv")
    fileInteriorRes = os.path.join(logFolder, "interiorTerms.csv")
    fileDataRes = os.path.join(logFolder, "dataTerms.csv")
    fileRegRes = os.path.join(logFolder, "regTerms.csv")
    fileGlobalLoss = os.path.join(logFolder, "log.csv")

    if ANNEALING:
        fileInteriorLossWeights = os.path.join(
            logFolder, "int_loss_weights.csv"
        )
        fileBoundaryLossWeights = os.path.join(
            logFolder, "bound_loss_weights.csv"
        )
        fileDataLossWeights = os.path.join(logFolder, "data_loss_weights.csv")
        fileRegLossWeights = os.path.join(logFolder, "reg_loss_weights.csv")

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
    ResDataName = [
        r"$\phi_{e}$",
        r"$\phi_{s,cath}$",
        r"C$_{s,an}$",
        r"C$_{s,ca}$",
    ]

    ResRegName = []

    passBoundaryCheck = True
    try:
        stepsBound, residualsBound, residualsBoundProportion = readResiduals(
            fileBoundaryRes
        )
        if ANNEALING:
            stepsBoundLossWeights, bound_loss_weights = read_loss_weights(
                fileBoundaryLossWeights
            )
    except:
        passBoundaryCheck = False
    if activeBound:
        if not len(ResBoundName) == residualsBound.shape[1] or (
            ANNEALING and not len(ResBoundName) == bound_loss_weights.shape[1]
        ):
            breakpoint()
            print(
                "Mismatch in boundary names and residuals. Update boundary residual names"
            )
            passBoundaryCheck = False

    passInteriorCheck = True
    try:
        stepsInt, residualsInt, residualsIntProportion = readResiduals(
            fileInteriorRes
        )
        if ANNEALING:
            stepsIntLossWeights, int_loss_weights = read_loss_weights(
                fileInteriorLossWeights
            )
    except:
        passInteriorCheck = False
    if activeInt:
        if not len(ResIntName) == residualsInt.shape[1] or (
            ANNEALING and not len(ResIntName) == int_loss_weights.shape[1]
        ):
            print(
                "Mismatch in physics names and residuals. Update physics residual names"
            )
            passInteriorCheck = False

    passDataCheck = True
    try:
        stepsData, residualsData, residualsDataProportion = readResiduals(
            fileDataRes
        )
        if ANNEALING:
            stepsDataLossWeights, data_loss_weights = read_loss_weights(
                fileDataLossWeights
            )
    except:
        passDataCheck = False
    if activeData:
        if not len(ResDataName) == residualsData.shape[1] or (
            ANNEALING and not len(ResDataName) == data_loss_weights.shape[1]
        ):
            print(
                "Mismatch in data names and residuals. Update data residual names"
            )
            passDataCheck = False

    passRegCheck = True
    try:
        stepsReg, residualsReg, residualsRegProportion = readResiduals(
            fileRegRes
        )
        if ANNEALING:
            stepsRegLossWeights, reg_loss_weights = read_loss_weights(
                fileRegLossWeights
            )
    except:
        passRegCheck = False
    if activeReg:
        if not len(ResRegName) == residualsReg.shape[1] or (
            ANNEALING and not len(ResRegName) == reg_loss_weights.shape[1]
        ):
            print(
                "Mismatch in reg names and residuals. Update Reg residual names"
            )
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
            stepsReg,
            residualsReg,
            ResRegName,
            title="Reg Losses",
            ax=axs[1, 1],
        )
    if not args.verbose:
        try:
            plt.savefig(
                os.path.join(figureFolder, logFolderFig, "residualsVal")
                + ".png"
            )
        except:
            print("Could not save residual value figure")

    if ANNEALING:
        fig, axs = plt.subplots(2, 2, figsize=(2 * 5, 2 * 5))
        if passBoundaryCheck:
            plotLossWeights(
                stepsBoundLossWeights,
                bound_loss_weights,
                ResBoundName,
                title="Boundary Losses",
                ax=axs[0, 0],
            )
        if passInteriorCheck:
            plotLossWeights(
                stepsIntLossWeights,
                int_loss_weights,
                ResIntName,
                title="Interior Losses",
                ax=axs[0, 1],
            )
        if passDataCheck:
            plotLossWeights(
                stepsDataLossWeights,
                data_loss_weights,
                ResDataName,
                title="Data Losses",
                ax=axs[1, 0],
            )
        if passRegCheck:
            plotLossWeights(
                stepsRegLossWeights,
                reg_loss_weights,
                ResRegName,
                title="Reg Losses",
                ax=axs[1, 1],
            )
        if not args.verbose:
            try:
                plt.savefig(
                    os.path.join(figureFolder, logFolderFig, "loss_weights")
                    + ".png"
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
                os.path.join(figureFolder, logFolderFig, "residualsProp")
                + ".png"
            )
        except:
            print("Could not save residual proportion figure")

    fig = plt.figure()
    if DYNAMIC_ATTENTION:
        plt.plot(
            globMSELoss[:, 0],
            globMSELoss[:, 2],
            linewidth=3,
            color="b",
            label="loss weighted",
        )
        plt.plot(
            globMSELoss[:, 0],
            globMSELoss[:, 3],
            linewidth=3,
            color="k",
            label="loss unweighted",
        )
    else:
        plt.plot(globMSELoss[:, 0], globMSELoss[:, 2], linewidth=3, color="k")

    pretty_labels("epoch", "Global loss", 14)
    if DYNAMIC_ATTENTION:
        pretty_legend()
    ax = plt.gca()
    ax.set_yscale("log")
    if not args.verbose:
        try:
            plt.savefig(
                os.path.join(figureFolder, logFolderFig, "globMSE_ep") + ".png"
            )
        except:
            print("Could not save global loss figure")

    fig = plt.figure()
    if DYNAMIC_ATTENTION:
        plt.plot(
            globMSELoss[:, 1],
            globMSELoss[:, 2],
            linewidth=3,
            color="b",
            label="loss weighted",
        )
        plt.plot(
            globMSELoss[:, 1],
            globMSELoss[:, 3],
            linewidth=3,
            color="k",
            label="loss unweighted",
        )
    else:
        plt.plot(globMSELoss[:, 1], globMSELoss[:, 2], linewidth=3, color="k")

    pretty_labels("epoch", "Global loss", 14)
    if DYNAMIC_ATTENTION:
        pretty_legend()
    ax = plt.gca()
    ax.set_yscale("log")
    if not args.verbose:
        try:
            plt.savefig(
                os.path.join(figureFolder, logFolderFig, "globMSE_step")
                + ".png"
            )
        except:
            print("Could not save global loss figure")

    if args.verbose:
        plt.show()


if __name__ == "__main__":
    # Read command line arguments
    args = argument.initArg()
    plot_res_var(args)
