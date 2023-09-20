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


def makeMovie(ntime, movieDir, movieName, prefix="im_"):
    fig = plt.figure()
    # initiate an empty  list of "plotted" images
    myimages = []
    # loops through available png:s
    for i in range(ntime):
        ## Read in picture
        fname = movieDir + "/" + prefix + str(i) + ".png"
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
        prettyLabels("", "", fontSize)
    else:
        prettyLabels("", "", fontSize, title=title)


def plot_probabilityMapDouble2D(
    model, minX, maxX, minY, maxY, nx=100, ny=100, minval=None, maxval=None
):
    x = np.linspace(minX, maxX, nx)
    y = np.linspace(minY, maxY, ny)
    sample = np.float64(np.zeros((nx, ny, 2)))
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


def pretty_bar_plot(
    xlabel1,
    yval,
    xlabel2=None,
    yerr=None,
    ymed=None,
    yerr_lower=None,
    yerr_upper=None,
    title=None,
    ylabel=None,
    bar_color=None,
    width=0.4,
    ylim=None,
    fontsize=14,
):
    if ylim is not None:
        assert len(ylim) == 2

    if xlabel2 is None:
        assert len(xlabel1) == len(yval)
        if yerr is not None:
            assert len(xlabel1) == len(yerr)

        fig = plt.figure(figsize=(len(xlabel1) * 2, 6))
        x = range(len(xlabel1))

        if bar_color is None:
            plt.bar(x, yval, width=width, align="center")
        else:
            plt.bar(x, yval, width=width, align="center", color=bar_color)
        if yerr is not None:
            plt.errorbar(
                x,
                yval,
                yerr,
                barsabove=True,
                capsize=5,
                elinewidth=3,
                fmt="none",
                color="k",
            )
        if (
            yerr_lower is not None
            and yerr_upper is not None
            and ymed is not None
        ):
            plt.errorbar(
                x,
                ymed,
                np.array(list(zip(yerr_lower, yerr_upper))).T,
                barsabove=True,
                capsize=5,
                elinewidth=3,
                fmt="none",
                color="k",
            )
        if ylabel is None:
            ylabel = ""
        if title is None:
            title = ""
        prettyLabels("", ylabel, title=title, fontsize=fontsize)
        ax = plt.gca()
        ax.set_xticks(x, xlabel1)

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
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

    else:
        # check yval
        assert len(yval) == len(xlabel2)
        assert len(yval[xlabel2[0]]) == len(xlabel1)

        if yerr is not None:
            assert len(yerr) == len(xlabel2)
            assert len(yerr[xlabel2[0]]) == len(xlabel1)
        elif (
            yerr_lower is not None
            and yerr_upper is not None
            and ymed is not None
        ):
            assert len(yerr_lower) == len(xlabel2)
            assert len(yerr_upper) == len(xlabel2)
            assert len(ymed) == len(xlabel2)
            assert len(yerr_lower[xlabel2[0]]) == len(xlabel1)
            assert len(yerr_upper[xlabel2[0]]) == len(xlabel1)
            assert len(ymed[xlabel2[0]]) == len(xlabel1)

        x = np.arange(len(xlabel1))  # the label locations
        width = width / len(xlabel2)  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(figsize=(len(xlabel1) * 2, 6))

        if yerr is not None:
            for (lab2, measurement), (lab2, measurement_err) in zip(
                yval.items(), yerr.items()
            ):
                offset = width * multiplier
                if bar_color is None:
                    rects = ax.bar(x + offset, measurement, width, label=lab2)
                else:
                    rects = ax.bar(
                        x + offset,
                        measurement,
                        width,
                        label=lab2,
                        color=bar_color,
                    )
                ax.errorbar(
                    x + offset,
                    measurement,
                    yerr=measurement_err,
                    barsabove=True,
                    capsize=5,
                    elinewidth=3,
                    fmt="none",
                    color="k",
                )
                multiplier += 1

        elif (
            yerr_lower is not None
            and yerr_upper is not None
            and ymed is not None
        ):
            for (
                (lab2, measurement),
                (lab2, measurement_err_lo),
                (lab2, measurement_err_hi),
                (lab2, measurement_med),
            ) in zip(
                yval.items(),
                yerr_lower.items(),
                yerr_upper.items(),
                ymed.items(),
            ):
                offset = width * multiplier
                if bar_color is None:
                    rects = ax.bar(x + offset, measurement, width, label=lab2)
                else:
                    rects = ax.bar(
                        x + offset,
                        measurement,
                        width,
                        label=lab2,
                        color=bar_color[xlabel2.index(lab2)],
                    )
                multiplier += 1
                ax.errorbar(
                    x + offset,
                    measurement_med,
                    np.array(
                        list(zip(measurement_err_lo, measurement_err_hi))
                    ).T,
                    barsabove=True,
                    capsize=5,
                    elinewidth=3,
                    fmt="none",
                    color="k",
                )

        else:
            for lab2, measurement in yval.items():
                offset = width * multiplier
                if bar_color is None:
                    rects = ax.bar(x + offset, measurement, width, label=lab2)
                else:
                    rects = ax.bar(
                        x + offset,
                        measurement,
                        width,
                        label=lab2,
                        color=bar_color,
                    )
                multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        if ylabel is None:
            ylabel = ""
        if title is None:
            title = ""
        prettyLabels("", ylabel, title=title, fontsize=fontsize)
        ax.set_xticks(x + width, xlabel1)

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

        if len(xlabel2) > 1:
            plotLegend()
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])


def line_cs_results(
    temp_pred,
    spac_pred,
    field_pred,
    time_stamps=[0, 200, 400],
    xlabel="",
    ylabel="",
    title=None,
    file_path_name=None,
    temp_dat=None,
    spac_dat=None,
    field_dat=None,
    verbose=False,
):
    plot_data = True
    if temp_dat is None or spac_dat is None or field_dat is None:
        plot_data = False

    if time_stamps is None or len(time_stamps) < 1:
        time_stamps = [0]

    # Line color
    color_stamp = []
    n_stamp = len(time_stamps)
    for istamp, stamp in enumerate(time_stamps):
        color_stamp.append(str(istamp * 0.6 / (n_stamp - 1)))

    # Find closest time to stamps
    ind_stamp_pred = []
    ind_stamp_dat = []
    for stamp in time_stamps:
        ind_stamp_pred.append(np.argmin(abs(temp_pred - float(stamp))))
        if plot_data:
            ind_stamp_dat.append(np.argmin(abs(temp_dat - float(stamp))))

    fig = plt.figure()
    if plot_data:
        for istamp in range(len(time_stamps)):
            plt.plot(
                spac_dat * 1e6,
                field_dat[ind_stamp_dat[istamp], :],
                "-.",
                linewidth=3,
                color=color_stamp[istamp],
            )

    for istamp, stamp in enumerate(time_stamps):
        plt.plot(
            spac_pred * 1e6,
            field_pred[ind_stamp_pred[istamp], :],
            linewidth=3,
            color=color_stamp[istamp],
            label=f"t = {stamp}s",
        )

    plotLegend()
    prettyLabels(xlabel, ylabel, 14, title=title)
    if not verbose and file_path_name is not None:
        plt.savefig(file_path_name)
        plt.close()


def line_phi_results(
    temp_pred,
    field_phie_pred,
    field_phis_c_pred,
    xlabel="time (s)",
    ylabel="(V)",
    title=None,
    file_path_name=None,
    temp_dat=None,
    field_phie_dat=None,
    field_phis_c_dat=None,
    verbose=False,
):
    plot_data = True
    if temp_dat is None or field_phie_dat is None or field_phis_c_dat is None:
        plot_data = False

    fig = plt.figure()
    if plot_data:
        plt.plot(temp_dat, field_phie_dat, "-.", linewidth=3, color="b")
        plt.plot(temp_dat, field_phis_c_dat, "-.", linewidth=3, color="k")

    plt.plot(
        temp_pred,
        field_phie_pred,
        linewidth=3,
        color="b",
        label=r"$\phi_{e}$",
    )
    plt.plot(
        temp_pred,
        field_phis_c_pred,
        linewidth=3,
        color="k",
        label=r"$\phi_{s,c}$",
    )

    plotLegend()
    prettyLabels(xlabel, ylabel, 14, title=title)
    if not verbose and file_path_name is not None:
        plt.savefig(file_path_name)
        plt.close()
