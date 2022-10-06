import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def prettyLabels(xlabel, ylabel, fontsize, title=None):
    plt.xlabel(
        xlabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.ylabel(
        ylabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    if not title == None:
        plt.title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    plt.grid(color="k", linestyle="-", linewidth=0.5)
    try:
        plt.tight_layout()
    except:
        print("Could not call tight_layout")
        pass


def plotLegend():
    fontsize = 16
    plt.legend()
    leg = plt.legend(
        prop={
            "family": "Times New Roman",
            "size": fontsize - 3,
            "weight": "bold",
        }
    )
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor("k")


def snapVizZslice(field, x, y, figureDir, figureName, title=None):
    fig, ax = plt.subplots(1)
    plt.imshow(
        np.transpose(field),
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=np.amin(field),
        vmax=np.amax(field),
        extent=[np.amin(x), np.amax(x), np.amax(y), np.amin(y)],
    )
    prettyLabels("x [m]", "y [m]", 16, title)
    plt.colorbar()
    fig.savefig(figureDir + "/" + figureName)
    plt.close(fig)
    return 0


def movieVizZslice(field, x, y, itime, movieDir, minVal=None, maxVal=None):
    fig, ax = plt.subplots(1)
    fontsize = 16
    if minVal == None:
        minVal = np.amin(field)
    if maxVal == None:
        maxVal = np.amax(field)
    plt.imshow(
        np.transpose(field),
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=minVal,
        vmax=maxVal,
        extent=[np.amin(x), np.amax(x), np.amax(y), np.amin(y)],
    )
    prettyLabels("x [m]", "y [m]", 16, "Snap Id = " + str(itime))
    plt.colorbar()
    fig.savefig(movieDir + "/im_" + str(itime) + ".png")
    plt.close(fig)
    return 0


def makeMovie(ntime, movieDir, movieName):
    fig = plt.figure()
    # initiate an empty  list of "plotted" images
    myimages = []
    # loops through available png:s
    for i in range(ntime):
        ## Read in picture
        fname = movieDir + "/im_" + str(i) + ".png"
        myimages.append(imageio.imread(fname))
    imageio.mimsave(movieName, myimages)
    return


def plotHist(field, xLabel, folder, filename):
    fig = plt.figure()
    plt.hist(field)
    fontsize = 18
    prettyLabels(xLabel, "bin count", fontsize)
    fig.savefig(folder + "/" + filename)


def plotContour(x, y, z, color):
    ax = plt.gca()
    X, Y = np.meshgrid(x, y)
    CS = ax.contour(
        X, Y, np.transpose(z), [0.001, 0.005, 0.01, 0.05], colors=color
    )
    h, _ = CS.legend_elements()
    return h[0]


def plotActiveSubspace(paramName, W, title=None):
    x = []
    for i, name in enumerate(paramName):
        x.append(i)
    fig = plt.figure()
    plt.bar(
        x,
        W,
        width=0.8,
        bottom=None,
        align="center",
        data=None,
        tick_label=paramName,
    )
    fontsize = 16
    if not title == None:
        plt.title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
        # ax.spines[axis].set_zorder(0)
    plt.grid(color="k", linestyle="-", linewidth=0.5)
    try:
        plt.tight_layout()
    except:
        print("Could not call tight_layout")
        pass


def axprettyLabels(ax, xlabel, ylabel, fontsize, title=None):
    ax.set_xlabel(
        xlabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    ax.set_ylabel(
        ylabel,
        fontsize=fontsize,
        fontweight="bold",
        fontname="Times New Roman",
    )
    if not title == None:
        ax.set_title(
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontname="Times New Roman",
        )
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontname("Times New Roman")
        tick.label1.set_fontweight("bold")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color("black")
    ax.grid(color="k", linestyle="-", linewidth=0.5)
    try:
        plt.tight_layout()
    except:
        print("Could not call tight_layout")
        pass


def axplotLegend(ax):
    fontsize = 16
    ax.legend()
    leg = ax.legend(
        prop={
            "family": "Times New Roman",
            "size": fontsize - 3,
            "weight": "bold",
        }
    )
    leg.get_frame().set_linewidth(2.0)
    leg.get_frame().set_edgecolor("k")


def plotTrainingLogs(trainingLoss, validationLoss):
    fig = plt.figure()
    plt.plot(trainingLoss, color="k", linewidth=3, label="train")
    plt.plot(validationLoss, "--", color="k", linewidth=3, label="test")
    prettyLabels("epoch", "loss", 14, title="model loss")
    plotLegend()


def plotScatter(
    dataX, dataY, freq, title=None, xfeat=None, yfeat=None, fontSize=14
):
    fig = plt.figure()
    if xfeat is None:
        xfeat = 0
    if yfeat is None:
        yfeat = 1

    plt.plot(dataX[0::freq], dataY[0::freq], "o", color="k", markersize=3)
    if title is None:
        # prettyLabels('feature '+str(xfeat),'feature '+str(yfeat),fontSize)
        prettyLabels("", "", fontSize)
    else:
        prettyLabels("", "", fontSize, title=title)


def plot_probabilityMapDouble2D(
    model, minX, maxX, minY, maxY, nx=100, ny=100, minval=None, maxval=None
):
    x = np.linspace(minX, maxX, nx)
    y = np.linspace(minY, maxY, ny)
    sample = np.float32(np.zeros((nx, ny, 2)))
    for i in range(nx):
        for j in range(ny):
            sample[i, j, 0] = x[i]
            sample[i, j, 1] = y[j]
    sample = np.reshape(sample, (nx * ny, 2))
    prob = np.exp(model.log_prob(sample))
    prob = np.reshape(prob, (nx, ny))

    if minval is None:
        minval = np.amin(prob)
    if maxval is None:
        maxval = np.amax(prob)

    fig = plt.figure()
    plt.imshow(
        np.transpose(prob),
        cmap=cm.jet,
        interpolation="bicubic",
        vmin=minval,
        vmax=maxval,
        extent=[np.amin(x), np.amax(x), np.amax(y), np.amin(y)],
        aspect="auto",
    )
    plt.gca().invert_yaxis()
    plt.colorbar()
    prettyLabels(
        "1st label", "2nd label", 20, title="Approximate Probability Map"
    )


def plot_fromLatentToData(model, nSamples, xfeat=None, yfeat=None):
    if xfeat is None:
        xfeat = 0
    if yfeat is None:
        yfeat = 1
    samples = model.distribution.sample(nSamples)
    print(samples.shape)
    x, _ = model.predict(samples)
    f, axes = plt.subplots(1, 2)
    axes[0].plot(
        samples[:, xfeat], samples[:, yfeat], "o", markersize=3, color="k"
    )
    axprettyLabels(
        axes[0],
        "feature " + str(xfeat),
        "feature " + str(yfeat),
        14,
        title="Prior",
    )
    axes[1].plot(x[:, xfeat], x[:, yfeat], "o", markersize=3, color="k")
    axprettyLabels(
        axes[1],
        "feature " + str(xfeat),
        "feature " + str(yfeat),
        14,
        title="Generated",
    )
