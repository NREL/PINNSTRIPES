import sys

import numpy as np
import tensorflow as tf
from thermo import *

tf.keras.backend.set_floatx("float64")

print("INFO: USING SIMPLE SPM MODEL")


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
            self.uocp = uocp_a_simp
            self.i0 = i0_a_simp_degradation_param
            self.ds = ds_a_fun_simp

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
            self.uocp = uocp_c_simp
            self.i0 = i0_c_simp
            self.ds = ds_c_fun_simp_degradation_param

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

        class Separator_IC:
            def __init__(self):
                self.ce = np.float64(1.2)

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
    from prettyPlot.plotting import plt, pretty_labels

    params = makeParams()

    print("rescalePhisCA = ", params["rescale_phis_c"])
    print("rescalePhie = ", params["rescale_phie"])
    print("rescaleCsCA = ", params["rescale_cs_c"])
    print("rescaleCsAN = ", params["rescale_cs_a"])

    # UOCP_a
    fig = plt.figure()
    cs = np.linspace(0, params["csanmax"], 100)
    u = params["Uocp_a"](cs, params["csanmax"])
    plt.plot(cs, u, color="k", linewidth=3)
    pretty_labels("cs", "U [V]", 14, r"U$_{ocp,an}$")

    # UOCP_c
    fig = plt.figure()
    cs = np.linspace(0, params["cscamax"], 100)
    u = params["Uocp_c"](cs, params["cscamax"])
    plt.plot(cs, u, color="k", linewidth=3)
    pretty_labels("cs", "U [V]", 14, r"U$_{ocp,ca}$")

    # I0_a
    fig = plt.figure()
    ce = np.linspace(0, 2 * params["ce0"], 100)
    i0 = params["i0_a"](
        params["csanmax"] / 2,
        ce,
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        np.ones(ce.shape).astype("float64"),
    )
    plt.plot(ce, np.array(i0), color="k", linewidth=3)
    pretty_labels("ce", "i", 14, r"I$_{0,a}$, cmax=csanmax/2")

    # I0_c
    fig = plt.figure()
    ce = np.linspace(0, 2 * params["ce0"], 100)
    i0 = params["i0_c"](
        params["cscamax"] / 2,
        ce,
        params["T"],
        params["alpha_c"],
        params["cscamax"],
        params["R"],
    )
    plt.plot(ce, i0, color="k", linewidth=3)
    pretty_labels("ce", "i", 14, r"I$_{0,c}$, cmax=csanmax/2")

    # DS_a
    fig = plt.figure()
    ds_a = params["D_s_a"](params["T"], params["R"])
    plt.plot(np.ones(100) * ds_a, color="k", linewidth=3)
    pretty_labels("", "Ds", 14, r"D$_{s,a}$")

    # DS_C
    fig = plt.figure()
    cs = np.linspace(0, params["cscamax"], 100)
    params["D_s_c"] = ds_c_fun_plot_simp
    ds = params["D_s_c"](cs, params["T"], params["R"], params["cscamax"])
    plt.plot(cs, ds, color="k", linewidth=3)
    pretty_labels("cs", "Ds", 14, r"D$_{s,c}$")

    plt.show()
