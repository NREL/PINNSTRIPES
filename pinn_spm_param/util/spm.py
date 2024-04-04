import sys

import keras
import numpy as np
import tensorflow as tf
from thermo import *

keras.backend.set_floatx("float64")

print("INFO: USING REALISTIC SPM MODEL")


def makeParams():
    class Degradation:
        def __init__(self):
            self.n_params = 2
            self.ind_i0_a = 0
            self.ind_ds_c = 1
            self.bounds = [[] for _ in range(self.n_params)]
            self.ref_vals = [0 for _ in range(self.n_params)]

            self.eff = np.float64(0.0)
            self.bounds[self.ind_i0_a] = [np.float64(0.5), np.float64(4)]
            self.bounds[self.ind_ds_c] = [np.float64(1.0), np.float64(10)]
            self.ref_vals[self.ind_i0_a] = np.float64(0.5)
            self.ref_vals[self.ind_ds_c] = np.float64(1.0)

    class Macroscopic:
        def __init__(self):
            self.F = np.float64(96485.3321e3)
            self.R = np.float64(8.3145e3)
            self.T = np.float64(303.15)
            self.T_const = np.float64(298.15)
            self.T_ref = np.float64(303.15)
            self.C = np.float64(-2.0)
            self.tmin = np.float64(0)
            self.tmax = np.float64(1350)
            self.rmin = np.float64(0)
            self.I = np.float64(1.89e-2 * self.C)

    class Anode:
        def __init__(self):
            self.thickness = np.float64(44 * 1e-6)
            self.solids = self.Anode_solids()
            self.A = np.float64(1.4e-3)
            self.alpha = np.float64(0.5)
            self.D50 = np.float64(8e-6)
            self.csmax = np.float64(30.53)
            self.uocp = uocp_a_fun
            self.i0 = i0_a_degradation_param_fun
            self.ds = ds_a_fun

        class Anode_solids:
            def __init__(self):
                self.eps = np.float64(0.5430727763)

    class Cathode:
        def __init__(self):
            self.thickness = np.float64(42 * 1e-6)
            self.A = np.float64(1.4e-3)
            self.solids = self.Cathode_solids()
            self.alpha = np.float64(0.5)
            self.D50 = np.float64(3.6e-6)
            self.csmax = np.float64(49.6)
            self.uocp = uocp_c_fun
            self.i0 = i0_c_fun
            self.ds = ds_c_degradation_param_fun

        class Cathode_solids:
            def __init__(self):
                self.eps = np.float64(0.47662)

    deg = Degradation()
    bat = Macroscopic()
    an = Anode()
    ca = Cathode()

    class IC:
        def __init__(self):
            self.an = self.Anode_IC()
            self.ca = self.Cathode_IC(self.an.cs)
            self.ce = np.float64(1.2)
            self.phie = -an.uocp(self.an.cs, an.csmax)
            self.phis_c = ca.uocp(self.ca.cs, ca.csmax) - an.uocp(
                self.an.cs, an.csmax
            )

        class Anode_IC:
            def __init__(self):
                self.ce = np.float64(1.2)
                self.cs = np.float64(0.91 * an.csmax)
                self.phis = np.float64(0.0)

        class Cathode_IC:
            def __init__(self, cs_a0):
                self.ce = np.float64(1.2)
                self.cs = np.float64(0.39 * ca.csmax)
                self.phis = ca.uocp(self.cs, ca.csmax) - an.uocp(
                    cs_a0, an.csmax
                )

    ic = IC()

    params = {}

    params = setParams(params, deg, bat, an, ca, ic)

    return params


if __name__ == "__main__":
    import os

    import pandas as pd
    from prettyPlot.plotting import plt, pretty_labels, pretty_legend

    params = makeParams()
    print("rescalePhisCA = ", params["rescale_phis_c"])
    print("rescalePhie = ", params["rescale_phie"])
    print("rescaleCsCA = ", params["rescale_cs_c"])
    print("rescaleCsAN = ", params["rescale_cs_a"])

    figureFolder = "Figures"
    refFolder = "../../Data/stateEqRef"

    os.makedirs(figureFolder, exist_ok=True)

    # UOCP_a
    fig = plt.figure()
    uocp_a_ref_file = os.path.join(refFolder, "anEeq.csv")
    uocp_a_ref_data = pd.read_csv(uocp_a_ref_file).to_numpy()
    x_ref = uocp_a_ref_data[:, 2]
    x_ref = x_ref[~np.isnan(x_ref)]
    cs_a_ref = x_ref * params["csanmax"]
    uocp_a_ref = uocp_a_ref_data[:, 3]
    uocp_a_ref = uocp_a_ref[~np.isnan(uocp_a_ref)]
    u = params["Uocp_a"](cs_a_ref, params["csanmax"])
    plt.plot(cs_a_ref, uocp_a_ref, "x", color="r", linewidth=3, label="Exp.")
    plt.plot(cs_a_ref, u, color="k", linewidth=3, label="Interp.")
    pretty_labels(r"Cs$_{an}$ [kmol/m$^3$]", "U [V]", 14, r"U$_{ocp,an}$")
    pretty_legend()
    plt.savefig(os.path.join(figureFolder, "uocp_a.png"))
    plt.close()
    gu = np.gradient(u, cs_a_ref)
    guref = np.gradient(uocp_a_ref, cs_a_ref)
    plt.plot(cs_a_ref, guref, "x", color="r", linewidth=3, label="Exp.")
    plt.plot(cs_a_ref, gu, color="k", linewidth=3, label="Interp.")
    pretty_labels(
        r"Cs$_{an}$ [kmol/m$^3$]", "", 14, r"$\nabla_{cs}$ U$_{ocp,an}$"
    )
    print(f"min grad gu = {np.amin(abs(gu))}")
    print(f"min grad guref = {np.amin(abs(guref))}")
    pretty_legend()
    plt.savefig(os.path.join(figureFolder, "guocp_a.png"))
    plt.close()

    # UOCP_c
    fig = plt.figure()
    uocp_c_ref_file = os.path.join(refFolder, "caEeq.csv")
    uocp_c_ref_data = pd.read_csv(uocp_c_ref_file).to_numpy()
    x_ref = uocp_c_ref_data[:, 0]
    cs_c_ref = x_ref * params["cscamax"]
    uocp_c_ref = uocp_c_ref_data[:, 1]
    u = params["Uocp_c"](cs_c_ref, params["cscamax"])
    plt.plot(cs_c_ref, uocp_c_ref, "x", color="r", linewidth=3, label="Exp.")
    plt.plot(cs_c_ref, u, color="k", linewidth=3, label="Interp.")
    pretty_labels(r"Cs$_{ca}$ [kmol/m$^3$]", "U [V]", 14, r"U$_{ocp,ca}$")
    pretty_legend()
    plt.savefig(os.path.join(figureFolder, "uocp_c.png"))
    plt.close()
    gu = np.gradient(u, cs_c_ref)
    guref = np.gradient(uocp_c_ref, cs_c_ref)
    plt.plot(cs_c_ref, guref, "x", color="r", linewidth=3, label="Exp.")
    plt.plot(cs_c_ref, gu, color="k", linewidth=3, label="Interp.")
    pretty_labels(
        r"Cs$_{ca}$ [kmol/m$^3$]", "", 14, r"$\nabla_{cs}$ U$_{ocp,ca}$"
    )
    pretty_legend()
    plt.savefig(os.path.join(figureFolder, "guocp_c.png"))
    plt.close()

    # I0_a
    fig = plt.figure()
    i0_a_ref_file = os.path.join(refFolder, "ani0.csv")
    i0_a_ref_data = pd.read_csv(i0_a_ref_file).to_numpy()
    x_surf_ref = i0_a_ref_data[:, 2]
    ce = np.ones(x_surf_ref.shape) * params["ce0"]
    cs_a_surf_ref = x_surf_ref * params["csanmax"]
    i0_a_ref = i0_a_ref_data[:, 3]
    i0 = params["i0_a"](
        cs_a_surf_ref,
        ce,
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        np.float64(1.0) * np.ones(cs_a_surf_ref.shape),
    )
    plt.plot(cs_a_surf_ref, i0_a_ref, "x", color="r", linewidth=3, label="Ref")
    plt.plot(cs_a_surf_ref, i0, color="k", linewidth=3, label="PINN")
    pretty_labels(
        "cs surf [kmol/m3]",
        "i",
        14,
        r"i$_{0,a}$, ce=ce$_{a,0}$, no degradation",
    )
    pretty_legend()
    plt.savefig(os.path.join(figureFolder, "i0_a.png"))
    plt.close()

    # I0_c
    fig = plt.figure()
    i0_c_ref_file = os.path.join(refFolder, "cai0.csv")
    i0_c_ref_data = pd.read_csv(i0_c_ref_file).to_numpy()
    x_surf_ref = i0_c_ref_data[:, 4]
    cs_c_surf_ref = x_surf_ref * params["cscamax"]
    ce = np.ones(x_surf_ref.shape) * params["ce_c0"]
    i0_c_ref = i0_c_ref_data[:, 5]
    i0 = params["i0_c"](
        cs_c_surf_ref,
        ce,
        params["T"],
        params["alpha_c"],
        params["cscamax"],
        params["R"],
    )
    plt.plot(cs_c_surf_ref, i0_c_ref, "x", color="r", linewidth=3, label="Ref")
    plt.plot(cs_c_surf_ref, i0, color="k", linewidth=3, label="PINN")
    pretty_labels("cs surf [kmol/m3]", "i", 14, r"i$_{0,c}$, ce = ce$_{c,0}$")
    pretty_legend()
    plt.savefig(os.path.join(figureFolder, "i0_c.png"))
    plt.close()

    # DS_a
    fig = plt.figure()
    ds_a_ref_file = os.path.join(refFolder, "anD.csv")
    ds_a_ref_data = pd.read_csv(ds_a_ref_file).to_numpy()
    T_ref = ds_a_ref_data[:, 1]
    ds_a_ref = ds_a_ref_data[:, 2]
    ds_a = params["D_s_a"](T_ref, params["R"])
    plt.plot(T_ref, ds_a_ref, "x", color="r", linewidth=3, label="Ref")
    plt.plot(T_ref, ds_a, color="k", linewidth=3, label="PINN")
    pretty_labels("T [K]", "Ds", 14, r"D$_{s,a}$")
    plt.savefig(os.path.join(figureFolder, "ds_a.png"))
    plt.close()

    # DS_C
    fig = plt.figure()
    ds_c_ref_file = os.path.join(refFolder, "caD.csv")
    ds_c_ref_data = pd.read_csv(ds_c_ref_file).to_numpy()
    x_ref = ds_c_ref_data[:, 4]
    cs_c_ref = x_ref * params["cscamax"]
    ds_c_ref = ds_c_ref_data[:, 5]
    deg = np.ones(cs_c_ref.shape)
    ds = params["D_s_c"](
        cs_c_ref, params["T"], params["R"], params["cscamax"], deg
    ).numpy()
    plt.plot(cs_c_ref, ds_c_ref, "x", color="r", linewidth=3, label="Ref")
    plt.plot(cs_c_ref, ds, color="k", linewidth=3, label="PINN")
    pretty_labels("cs", "Ds", 14, r"D$_{s,c}$, no degradation")
    pretty_legend()
    plt.savefig(os.path.join(figureFolder, "ds_c.png"))
    plt.close()

    plt.show()
