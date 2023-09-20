import numpy as np
from plotsUtil import *


def plotField_all(fieldList, fieldList2, xList, label, label2, name):
    x_a = xList[0]
    x_s = xList[1]
    x_c = xList[2]

    field_a = fieldList[0]
    field_s = fieldList[1]
    field_c = fieldList[2]

    plotField2 = False
    if not fieldList2 == None:
        field2_a = fieldList2[0]
        field2_s = fieldList2[1]
        field2_c = fieldList2[2]
        plotField2 = True

    nt = field_a.shape[0]

    # Anode
    fig = plt.figure()
    for i in range(10):
        if i == 0:
            plt.plot(
                x_a,
                field_a[i * nt // 10, :],
                linewidth=3,
                color="k",
                label=label,
            )
            if plotField2:
                plt.plot(
                    x_a,
                    field2_a[i * nt // 10, :],
                    "x",
                    color="b",
                    label=label2,
                )
        else:
            plt.plot(x_a, field_a[i * nt // 10, :], linewidth=3, color="k")
            if plotField2:
                plt.plot(x_a, field2_a[i * nt // 10, :], "x", color="b")
    prettyLabels("x", name, 14, title="anode")
    # Separator
    fig = plt.figure()
    for i in range(10):
        if i == 0:
            plt.plot(
                x_s,
                field_s[i * nt // 10, :],
                linewidth=3,
                color="k",
                label=label,
            )
            if plotField2:
                plt.plot(
                    x_s,
                    field2_s[i * nt // 10, :],
                    "x",
                    color="b",
                    label=label2,
                )
        else:
            plt.plot(x_s, field_s[i * nt // 10, :], linewidth=3, color="k")
            if plotField2:
                plt.plot(x_s, field2_s[i * nt // 10, :], "x", color="b")
    prettyLabels("x", name, 14, title="separator")
    # Cathode
    fig = plt.figure()
    for i in range(10):
        if i == 0:
            plt.plot(
                x_c,
                field_c[i * nt // 10, :],
                linewidth=3,
                color="k",
                label=label,
            )
            if plotField2:
                plt.plot(
                    x_c,
                    field2_c[i * nt // 10, :],
                    "x",
                    color="b",
                    label=label2,
                )
        else:
            plt.plot(x_c, field_c[i * nt // 10, :], linewidth=3, color="k")
            if plotField2:
                plt.plot(x_c, field2_c[i * nt // 10, :], "x", color="b")
    prettyLabels("x", name, 14, title="cathode")

    return


def plotField_electrode(fieldList, fieldList2, xList, label, label2, name):
    x_a = xList[0]
    x_c = xList[1]

    field_a = fieldList[0]
    field_c = fieldList[1]

    plotField2 = False
    if not fieldList2 == None:
        field2_a = fieldList2[0]
        field2_c = fieldList2[1]
        plotField2 = True

    nt = field_a.shape[0]

    # Anode
    fig = plt.figure()
    for i in range(10):
        if i == 0:
            plt.plot(
                x_a,
                field_a[i * nt // 10, :],
                linewidth=3,
                color="k",
                label=label,
            )
            if plotField2:
                plt.plot(
                    x_a,
                    field2_a[i * nt // 10, :],
                    "x",
                    color="b",
                    label=label2,
                )
        else:
            plt.plot(x_a, field_a[i * nt // 10, :], linewidth=3, color="k")
            if plotField2:
                plt.plot(x_a, field2_a[i * nt // 10, :], "x", color="b")
    prettyLabels("x", name, 14, title="anode")
    # Cathode
    fig = plt.figure()
    for i in range(10):
        if i == 0:
            plt.plot(
                x_c,
                field_c[i * nt // 10, :],
                linewidth=3,
                color="k",
                label=label,
            )
            if plotField2:
                plt.plot(
                    x_c,
                    field2_c[i * nt // 10, :],
                    "x",
                    color="b",
                    label=label2,
                )
        else:
            plt.plot(x_c, field_c[i * nt // 10, :], linewidth=3, color="k")
            if plotField2:
                plt.plot(x_c, field2_c[i * nt // 10, :], "x", color="b")
    prettyLabels("x", name, 14, title="cathode")

    return


def plotField_single(
    field, field2, x, label, label2, name, component, nlines=10
):
    plotField2 = False
    if not field2 is None:
        plotField2 = True

    nt = field.shape[0]

    fig = plt.figure()
    for i in range(nlines):
        if i == 0:
            plt.plot(
                x,
                field[(i + 1) * nt // nlines - 1, :],
                linewidth=1,
                color="k",
                label=label,
            )
            if plotField2:
                plt.plot(
                    x,
                    field2[(i + 1) * nt // nlines - 1, :],
                    "x",
                    color="b",
                    label=label2,
                )
        else:
            plt.plot(
                x, field[(i + 1) * nt // nlines - 1, :], linewidth=1, color="k"
            )
            if plotField2:
                plt.plot(
                    x, field2[(i + 1) * nt // nlines - 1, :], "x", color="b"
                )
    plotLegend()
    prettyLabels("x", name, 14, title=component)

    return


def plotData(
    listDatax,
    listData,
    tmax,
    listCBLabel,
    listTitle,
    listXAxisName=None,
    vminList=None,
    vmaxList=None,
    globalTitle=None,
):
    lim = -1
    lim_vmax_t = -1
    lim_vmax_x = -1
    lim_plot = -1
    fig, axs = plt.subplots(1, len(listData), figsize=(len(listData) * 3, 4))
    if len(listData) == 1:
        i_dat = 0
        data = listData[i_dat]
        data_x = np.squeeze(listDatax[i_dat])
        if vminList == None:
            vmin = np.nanmin(data[:lim, :])
        else:
            vmin = vminList[i_dat]
        if vmaxList == None:
            vmax = np.nanmax(data[:lim, :])
        else:
            vmax = vmaxList[i_dat]
        im = axs.imshow(
            data[:lim, :],
            cmap=cm.viridis,
            interpolation="bicubic",
            vmin=vmin,
            vmax=vmax,
            extent=[0, data_x[-1], tmax, 0],
            aspect="auto",
        )
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="10%", pad=0.2)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(listCBLabel[i_dat])
        ax = cbar.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(
            family="times new roman", weight="bold", size=14
        )
        text.set_font_properties(font)
        if listXAxisName is None:
            if i_dat == 0:
                axprettyLabels(axs, "x", "t [s]", 12, listTitle[i_dat])
            else:
                axprettyLabels(axs, "x", "", 12, listTitle[i_dat])
        else:
            if i_dat == 0:
                axprettyLabels(
                    axs, listXAxisName[i_dat], "t [s]", 12, listTitle[i_dat]
                )
            else:
                axprettyLabels(
                    axs, listXAxisName[i_dat], "", 12, listTitle[i_dat]
                )

        ax.set_xticks([])  # values
        ax.set_xticklabels([])  # labels
        if not i_dat == 0:
            ax.set_yticks([])  # values
            ax.set_yticklabels([])  # labels
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_family("serif")
            l.set_fontsize(12)
    else:
        for i_dat in range(len(listData)):
            data = listData[i_dat]
            data_x = np.squeeze(listDatax[i_dat])
            if vminList == None:
                vmin = np.nanmin(data[:lim, :])
            else:
                vmin = vminList[i_dat]
            if vmaxList == None:
                vmax = np.nanmax(data[:lim, :])
            else:
                vmax = vmaxList[i_dat]
            im = axs[i_dat].imshow(
                data[:lim, :],
                cmap=cm.viridis,
                interpolation="bicubic",
                vmin=vmin,
                vmax=vmax,
                extent=[0, data_x[-1], tmax, 0],
                aspect="auto",
            )
            divider = make_axes_locatable(axs[i_dat])
            cax = divider.append_axes("right", size="10%", pad=0.2)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(listCBLabel[i_dat])
            ax = cbar.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(
                family="times new roman", weight="bold", size=14
            )
            text.set_font_properties(font)
            if listXAxisName is None:
                if i_dat == 0:
                    axprettyLabels(
                        axs[i_dat], "x", "t [s]", 12, listTitle[i_dat]
                    )
                else:
                    axprettyLabels(axs[i_dat], "x", "", 12, listTitle[i_dat])
            else:
                if i_dat == 0:
                    axprettyLabels(
                        axs[i_dat],
                        listXAxisName[i_dat],
                        "t [s]",
                        12,
                        listTitle[i_dat],
                    )
                else:
                    axprettyLabels(
                        axs[i_dat],
                        listXAxisName[i_dat],
                        "",
                        12,
                        listTitle[i_dat],
                    )
            axs[i_dat].set_xticks([])  # values
            axs[i_dat].set_xticklabels([])  # labels
            if not i_dat == 0:
                axs[i_dat].set_yticks([])  # values
                axs[i_dat].set_yticklabels([])  # labels
            for l in cbar.ax.yaxis.get_ticklabels():
                l.set_weight("bold")
                l.set_family("serif")
                l.set_fontsize(12)

    if not globalTitle is None:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(
            globalTitle,
            fontsize=14,
            fontweight="bold",
            fontname="Times New Roman",
        )


def plotCollWeights(
    listDatax,
    listDatat,
    listData,
    tmax,
    listTitle,
    listXAxisName=None,
    vminList=None,
    vmaxList=None,
    globalTitle=None,
):
    lim = -1
    lim_vmax_t = -1
    lim_vmax_x = -1
    lim_plot = -1
    fig, axs = plt.subplots(1, len(listData), figsize=(len(listData) * 3, 4))
    if len(listData) == 1:
        i_dat = 0
        data = listData[i_dat]
        data_x = np.squeeze(listDatax[i_dat])
        data_t = np.squeeze(listDatat[i_dat])
        if vminList == None:
            vmin = np.nanmin(data)
        else:
            try:
                vmin = vminList[i_dat]
            except:
                vmin = vminList[0]
        if vmaxList == None:
            vmax = np.nanmax(data) + 1e-10
        else:
            try:
                vmax = vmaxList[i_dat] + 1e-10
            except:
                vmax = vmaxList[0] + 1e-10
        cm = plt.cm.get_cmap("viridis")
        sc = axs.scatter(
            data_x, data_t, c=data, vmin=vmin, vmax=vmax, s=20, cmap=cm
        )
        axs.invert_yaxis()
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="10%", pad=0.2)
        cbar = fig.colorbar(sc, cax=cax)
        ax = cbar.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(
            family="times new roman", weight="bold", size=14
        )
        text.set_font_properties(font)
        if listXAxisName is None:
            if i_dat == 0:
                axprettyLabels(axs, "x", "t [s]", 12, listTitle[i_dat])
            else:
                axprettyLabels(axs, "x", "", 12, listTitle[i_dat])
        else:
            if i_dat == 0:
                axprettyLabels(
                    axs, listXAxisName[i_dat], "t [s]", 12, listTitle[i_dat]
                )
            else:
                axprettyLabels(
                    axs, listXAxisName[i_dat], "", 12, listTitle[i_dat]
                )

        ax.set_xticks([])  # values
        ax.set_xticklabels([])  # labels
        if not i_dat == 0:
            ax.set_yticks([])  # values
            ax.set_yticklabels([])  # labels
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_family("serif")
            l.set_fontsize(12)
    else:
        for i_dat in range(len(listData)):
            data = listData[i_dat]
            data_x = np.squeeze(listDatax[i_dat])
            data_t = np.squeeze(listDatat[i_dat])
            if vminList == None:
                vmin = np.nanmin(data)
            else:
                try:
                    vmin = vminList[i_dat]
                except:
                    vmin = vminList[0]
            if vmaxList == None:
                vmax = np.nanmax(data) + 1e-10
            else:
                try:
                    vmax = vmaxList[i_dat] + 1e-10
                except:
                    vmax = vmaxList[0] + 1e-10
            cm = plt.cm.get_cmap("viridis")
            sc = axs[i_dat].scatter(
                data_x, data_t, c=data, vmin=vmin, vmax=vmax, s=20, cmap=cm
            )
            axs[i_dat].invert_yaxis()
            divider = make_axes_locatable(axs[i_dat])
            cax = divider.append_axes("right", size="10%", pad=0.2)
            cbar = fig.colorbar(sc, cax=cax)
            ax = cbar.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(
                family="times new roman", weight="bold", size=14
            )
            text.set_font_properties(font)
            if listXAxisName is None:
                if i_dat == 0:
                    axprettyLabels(
                        axs[i_dat], "x", "t [s]", 12, listTitle[i_dat]
                    )
                else:
                    axprettyLabels(axs[i_dat], "x", "", 12, listTitle[i_dat])
            else:
                if i_dat == 0:
                    axprettyLabels(
                        axs[i_dat],
                        listXAxisName[i_dat],
                        "t [s]",
                        12,
                        listTitle[i_dat],
                    )
                else:
                    axprettyLabels(
                        axs[i_dat],
                        listXAxisName[i_dat],
                        "",
                        12,
                        listTitle[i_dat],
                    )

            if not i_dat == 0:
                axs[i_dat].set_yticks([])
                axs[i_dat].set_yticklabels([])
            for l in cbar.ax.yaxis.get_ticklabels():
                l.set_weight("bold")
                l.set_family("serif")
                l.set_fontsize(12)

        if not globalTitle is None:
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle(
                globalTitle,
                fontsize=14,
                fontweight="bold",
                fontname="Times New Roman",
            )
