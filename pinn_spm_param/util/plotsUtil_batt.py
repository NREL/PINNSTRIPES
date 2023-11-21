import numpy as np
from prettyPlot.plotsUtil import (
    cm,
    make_axes_locatable,
    matplotlib,
    plt,
    pretty_labels,
    pretty_legend,
)


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

    pretty_legend()
    pretty_labels(xlabel, ylabel, 14, title=title)
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

    pretty_legend()
    pretty_labels(xlabel, ylabel, 14, title=title)
    if not verbose and file_path_name is not None:
        plt.savefig(file_path_name)
        plt.close()


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
    pretty_labels("x", name, 14, title="anode")
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
    pretty_labels("x", name, 14, title="separator")
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
    pretty_labels("x", name, 14, title="cathode")

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
    pretty_labels("x", name, 14, title="anode")
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
    pretty_labels("x", name, 14, title="cathode")

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
    pretty_legend()
    pretty_labels("x", name, 14, title=component)

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
                pretty_labels("x", "t [s]", 12, listTitle[i_dat], ax=axs)
            else:
                pretty_labels("x", "", 12, listTitle[i_dat], ax=axs)
        else:
            if i_dat == 0:
                pretty_labels(
                    listXAxisName[i_dat], "t [s]", 12, listTitle[i_dat], ax=axs
                )
            else:
                pretty_labels(
                    listXAxisName[i_dat], "", 12, listTitle[i_dat], ax=axs
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
                    pretty_labels(
                        "x", "t [s]", 12, listTitle[i_dat], ax=axs[i_dat]
                    )
                else:
                    pretty_labels("x", "", 12, listTitle[i_dat], ax=axs[i_dat])
            else:
                if i_dat == 0:
                    pretty_labels(
                        listXAxisName[i_dat],
                        "t [s]",
                        12,
                        listTitle[i_dat],
                        ax=axs[i_dat],
                    )
                else:
                    pretty_labels(
                        listXAxisName[i_dat],
                        "",
                        12,
                        listTitle[i_dat],
                        ax=axs[i_dat],
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
                pretty_labels("x", "t [s]", 12, listTitle[i_dat], ax=axs)
            else:
                pretty_labels("x", "", 12, listTitle[i_dat], ax=axs)
        else:
            if i_dat == 0:
                pretty_labels(
                    listXAxisName[i_dat], "t [s]", 12, listTitle[i_dat], ax=axs
                )
            else:
                pretty_labels(
                    listXAxisName[i_dat], "", 12, listTitle[i_dat], ax=axs
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
                    pretty_labels(
                        "x", "t [s]", 12, listTitle[i_dat], ax=axs[i_dat]
                    )
                else:
                    pretty_labels("x", "", 12, listTitle[i_dat], ax=axs[i_dat])
            else:
                if i_dat == 0:
                    pretty_labels(
                        listXAxisName[i_dat],
                        "t [s]",
                        12,
                        listTitle[i_dat],
                        ax=axs[i_dat],
                    )
                else:
                    pretty_labels(
                        listXAxisName[i_dat],
                        "",
                        12,
                        listTitle[i_dat],
                        ax=axs[i_dat],
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
