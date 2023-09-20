import sys

import numpy as np
import tensorflow as tf
from uocp_cs import uocp_a_fun_x, uocp_c_fun_x

tf.keras.backend.set_floatx("float64")


def uocp_a_simp(cs_a, csanmax):
    x = cs_a / csanmax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return np.float64(0.2) - np.float64(0.2) * x


def uocp_a_fun(cs_a, csanmax):
    x = cs_a / csanmax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return uocp_a_fun_x(x)


def uocp_c_fun(cs_c, cscamax):
    x = cs_c / cscamax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return uocp_c_fun_x(x)


def uocp_c_simp(cs_c, cscamax):
    x = cs_c / cscamax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return np.float64(5.0) - np.float64(1.4) * x


def i0_a_fun(cs_a_max, ce, T, alpha, csanmax, R):
    return (
        np.float64(2.5)
        * np.float64(0.27)
        * tf.exp(
            np.float64(
                (-30.0e6 / R)
                * (np.float64(1.0) / T - np.float64(1.0) / np.float64(303.15))
            )
        )
        * tf.math.maximum(ce, np.float64(0.0)) ** alpha
        * tf.math.maximum(csanmax - cs_a_max, np.float64(0.0)) ** alpha
        * tf.math.maximum(cs_a_max, np.float64(0.0))
        ** (np.float64(1.0) - alpha)
    )


def i0_a_degradation_param_fun(
    cs_a_max, ce, T, alpha, csanmax, R, degradation_param
):
    i0_a_nodeg = i0_a_fun(cs_a_max, ce, T, alpha, csanmax, R)
    return tf.reshape(degradation_param, tf.shape(i0_a_nodeg)) * i0_a_nodeg


def i0_a_simp(cs_a_max, ce, T, alpha, csanmax, R):
    return np.float64(2.0) * np.ones(ce.shape, dtype="float64")


def i0_a_simp_degradation_param(
    cs_a_max, ce, T, alpha, csanmax, R, degradation_param
):
    return (
        np.float64(2.0)
        * tf.reshape(degradation_param, tf.shape(ce))
        * np.ones(ce.shape, dtype="float64")
    )


def i0_c_fun(cs_c_max, ce, T, alpha, cscamax, R):
    x = cs_c_max / cscamax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    return (
        np.float64(9.0)
        * (
            np.float64(1.650452829641290e01) * x**5
            - np.float64(7.523567141488800e01) * x**4
            + np.float64(1.240524690073040e02) * x**3
            - np.float64(9.416571081287610e01) * x**2
            + np.float64(3.249768821737960e01) * x
            - np.float64(3.585290065824760e00)
        )
        * tf.math.maximum(ce / np.float64(1.2), np.float64(0.0)) ** alpha
        * tf.exp(
            (np.float64(-30.0e6) / R)
            * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
        )
    )


def i0_c_simp(cs_c_max, ce, T, alpha, cscamax, R):
    return np.float64(3.0) * np.ones(ce.shape, dtype="float64")


def ds_a_fun(T, R):
    return np.float64(3.0e-14) * tf.exp(
        (np.float64(-30.0e6) / R)
        * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
    )


def grad_ds_a_cs_a(T, R):
    return np.float64(0.0)


def ds_a_fun_simp(T, R):
    return np.float64(3.0e-14)


def ds_c_fun(cs_c, T, R, cscamax):
    x = cs_c / cscamax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    power = (
        -np.float64(2.509010843479270e02) * x**10
        + np.float64(2.391026725259970e03) * x**9
        - np.float64(4.868420267611360e03) * x**8
        - np.float64(8.331104102921070e01) * x**7
        + np.float64(1.057636028329000e04) * x**6
        - np.float64(1.268324548348120e04) * x**5
        + np.float64(5.016272167775530e03) * x**4
        + np.float64(9.824896659649480e02) * x**3
        - np.float64(1.502439339070900e03) * x**2
        + np.float64(4.723709304247700e02) * x
        - np.float64(6.526092046397090e01)
    )
    return (
        np.float64(1.5)
        * (np.float64(1.5) * np.float64(10.0) ** (power))
        * tf.exp(
            (np.float64(-30.0e6) / R)
            * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
        )
    )


def grad_ds_c_cs_c(cs_c, T, R, cscamax):
    return (
        np.float64(2.25)
        * np.float64(10.0)
        ** (
            -np.float64(250.901084347927) * cs_c**10 / cscamax**10
            + np.float64(2391.02672525997) * cs_c**9 / cscamax**9
            - np.float64(4868.42026761136) * cs_c**8 / cscamax**8
            - np.float64(83.3110410292107) * cs_c**7 / cscamax**7
            + np.float64(10576.36028329) * cs_c**6 / cscamax**6
            - np.float64(12683.2454834812) * cs_c**5 / cscamax**5
            + np.float64(5016.27216777553) * cs_c**4 / cscamax**4
            + np.float64(982.489665964948) * cs_c**3 / cscamax**3
            - np.float64(1502.4393390709) * cs_c**2 / cscamax**2
            + np.float64(472.37093042477) * cs_c / cscamax
            - np.float64(65.2609204639709)
        )
        * (
            -np.float64(5777.21096635578) * cs_c**9 / cscamax**10
            + np.float64(49549.8824508058) * cs_c**8 / cscamax**9
            - np.float64(89679.615477056) * cs_c**7 / cscamax**8
            - np.float64(1342.81532808973) * cs_c**6 / cscamax**7
            + np.float64(146117.817158627) * cs_c**5 / cscamax**6
            - np.float64(146021.259905239) * cs_c**4 / cscamax**5
            + np.float64(46201.5740636835) * cs_c**3 / cscamax**4
            + np.float64(6786.79817661477) * cs_c**2 / cscamax**3
            - np.float64(6918.98885054496) * cs_c / cscamax**2
            + np.float64(1087.6742627598) / cscamax
        )
        * np.exp(
            -np.float64(30000000.0)
            * (-np.float64(0.0032986970146792) + np.float64(1.0) / T)
            / R
        )
    )


def ds_c_degradation_param_fun(cs_c, T, R, cscamax, degradation_param):
    ds_c_nodeg = ds_c_fun(cs_c, T, R, cscamax)
    return tf.reshape(degradation_param, tf.shape(ds_c_nodeg)) * ds_c_nodeg


def ds_c_fun_simp(cs_c, T, R, cscamax):
    return np.float64(3.5e-15) * np.ones(cs_c.shape, dtype="float64")


def ds_c_fun_plot(cs_c, T, R, cscamax):
    x = cs_c / cscamax
    x = tf.clip_by_value(x, np.float64(0.0), np.float64(1.0))
    power = (
        -np.float64(2.509010843479270e02) * x**10
        + np.float64(2.391026725259970e03) * x**9
        - np.float64(4.868420267611360e03) * x**8
        - np.float64(8.331104102921070e01) * x**7
        + np.float64(1.057636028329000e04) * x**6
        - np.float64(1.268324548348120e04) * x**5
        + np.float64(5.016272167775530e03) * x**4
        + np.float64(9.824896659649480e02) * x**3
        - np.float64(1.502439339070900e03) * x**2
        + np.float64(4.723709304247700e02) * x
        - np.float64(6.526092046397090e01)
    )
    try:
        return (
            np.float64(1.5)
            * (np.float64(1.5) * np.float64(10.0) ** (power))
            * tf.exp(
                (np.float64(-30.0e6) / R)
                * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
            )
        )
    except tf.python.framework.errors_impl.InvalidArgumentError:
        return (
            np.float64(1.5)
            * (np.float64(1.5) * np.float64(10.0) ** (power))
            * np.exp(
                (np.float64(-30.0e6) / R)
                * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
            )
        )


def ds_c_fun_plot_simp(cs_c, T, R, cscamax):
    return np.float64(3.5e-15) * np.ones(cs_c.shape, dtype="float64")


def ds_c_fun_simp_degradation_param(cs_c, T, R, cscamax, degradation_param):
    return (
        np.float64(3.5e-15)
        * tf.reshape(degradation_param, tf.shape(cs_c))
        * np.ones(cs_c.shape, dtype="float64")
    )


def sigma_c_fun(eps_s_c, eps_cbd_c):
    return np.float64(10.0) * (eps_s_c + eps_cbd_c)


def sigma_a_fun(eps_s_a, eps_cbd_a):
    return np.float64(10.0) * (eps_s_a + eps_cbd_a)


def de_fun(ce, T):
    power = (
        -np.float64(0.5688226)
        - np.float64(1607.003)
        / (T - (-np.float64(24.83763) + np.float64(64.07366) * ce))
        + (
            -np.float64(0.8108721)
            + np.float64(475.291)
            / (T - (-np.float64(24.83763) + np.float64(64.07366) * ce))
        )
        * ce
        + (
            -np.float64(0.005192312)
            - np.float64(33.43827)
            / (T - (-np.float64(24.83763) + np.float64(64.07366) * ce))
        )
        * ce
        * ce
    )
    return np.float64(0.0001) * tf.math.pow(np.float64(10.0), power)


def de_fun_simp(ce, T):
    return np.float64(1.8e-10) * np.ones(ce.shape, dtype="float64")


def ke_fun(ce, T):
    return ce * (
        (
            np.float64(0.0001909446) * T**2
            - np.float64(0.08038545) * T
            + np.float64(9.00341)
        )
        + ce
        * (
            -np.float64(0.00000002887587) * T**4
            + np.float64(0.00003483638) * T**3
            - np.float64(0.01583677) * T**2
            + np.float64(3.195295) * T
            - np.float64(241.4638)
        )
        + ce**2
        * (
            np.float64(0.00000001653786) * T**4
            - np.float64(0.0000199876) * T**3
            + np.float64(0.009071155) * T**2
            - np.float64(1.828064) * T
            + np.float64(138.0976)
        )
        + ce**3
        * (
            -np.float64(0.000000002791965) * T**4
            + np.float64(0.000003377143) * T**3
            - np.float64(0.001532707) * T**2
            + np.float64(0.3090003) * T
            - np.float64(23.35671)
        )
    )


def ke_fun_simp(ce, T):
    return np.float64(0.6) * np.ones(ce.shape, dtype="float64")


def dlnf_dce_fun(ce, T):
    return (
        np.float64(0.54) * ce**2 * tf.exp(np.float64(329.0) / T)
        - np.float64(0.00225) * ce * tf.exp(np.float64(1360.0) / T)
        + np.float64(0.341) * tf.exp(np.float64(261.0) / T)
    )


def dlnf_dce_fun_simp(ce, T):
    return np.float64(8.0) * np.ones(ce.shape, dtype="float64")


def t0_fun(ce, T):
    return (
        (
            np.float64(-0.0000002876102) * T**2
            + np.float64(0.0002077407) * T
            - np.float64(0.03881203)
        )
        * ce**2
        + (
            np.float64(0.000001161463) * T**2
            - np.float64(0.00086825) * T
            + np.float64(0.1777266)
        )
        * ce
        + (
            -np.float64(0.0000006766258) * T**2
            + np.float64(0.0006389189) * T
            + np.float64(0.3091761)
        )
    )


def t0_fun_simp(ce, T):
    return np.float64(0.46) * np.ones(ce.shape, dtype="float64")


def phie0_fun(i0_a, j_a, F, R, T, Uocp_a0):
    return -j_a * (F / i0_a) * (R * T / F) - Uocp_a0


def phis_c0_fun(i0_a, j_a, F, R, T, Uocp_a0, j_c, i0_c, Uocp_c0):
    phie0 = phie0_fun(i0_a, j_a, F, R, T, Uocp_a0)
    return j_c * (F / i0_c) * (R * T / F) + Uocp_c0 + phie0


def setParams(params, deg, bat, an, sep, ca, ic):
    # Parametric domain
    params["deg_i0_a_min"] = deg.bounds[deg.ind_i0_a][0]
    params["deg_i0_a_max"] = deg.bounds[deg.ind_i0_a][1]
    params["deg_ds_c_min"] = deg.bounds[deg.ind_ds_c][0]
    params["deg_ds_c_max"] = deg.bounds[deg.ind_ds_c][1]

    params["param_eff"] = deg.eff
    params["deg_i0_a_ref"] = deg.ref_vals[deg.ind_i0_a]
    params["deg_ds_c_ref"] = deg.ref_vals[deg.ind_ds_c]
    params["deg_i0_a_min_eff"] = (
        params["deg_i0_a_ref"]
        + (params["deg_i0_a_min"] - params["deg_i0_a_ref"])
        * params["param_eff"]
    )
    params["deg_i0_a_max_eff"] = (
        params["deg_i0_a_ref"]
        + (params["deg_i0_a_max"] - params["deg_i0_a_ref"])
        * params["param_eff"]
    )
    params["deg_ds_c_min_eff"] = (
        params["deg_ds_c_ref"]
        + (params["deg_ds_c_min"] - params["deg_ds_c_ref"])
        * params["param_eff"]
    )
    params["deg_ds_c_max_eff"] = (
        params["deg_ds_c_ref"]
        + (params["deg_ds_c_max"] - params["deg_ds_c_ref"])
        * params["param_eff"]
    )
    # Domain
    params["tmin"] = bat.tmin
    params["tmax"] = bat.tmax
    params["rmin"] = bat.rmin

    # Params fixed
    params["A_a"] = an.A
    params["A_c"] = ca.A
    params["F"] = bat.F
    params["R"] = bat.R
    params["T"] = bat.T
    params["C"] = bat.C
    params["I_discharge"] = bat.I

    params["alpha_a"] = an.alpha
    params["alpha_c"] = ca.alpha

    # Params to fit
    params["Rs_a"] = an.D50 / np.float64(2.0)
    params["Rs_c"] = ca.D50 / np.float64(2.0)
    params["rescale_R"] = np.float64(max(params["Rs_a"], params["Rs_c"]))
    # current
    params["csanmax"] = an.csmax
    params["cscamax"] = ca.csmax
    params["rescale_T"] = np.float64(max(bat.tmax, 1e-16))

    # Typical variables magnitudes
    params["mag_cs_a"] = np.float64(25)  # OK
    params["mag_cs_c"] = np.float64(32.5)  # OK
    params["mag_phis_c"] = np.float64(4.25)  # OK
    params["mag_phie"] = np.float64(0.15)  # OK
    params["mag_ce"] = np.float64(1.2)  # OK

    # FUNCTIONS
    params["Uocp_a"] = an.uocp
    params["Uocp_c"] = ca.uocp
    params["i0_a"] = an.i0
    params["i0_c"] = ca.i0
    params["D_s_a"] = an.ds
    params["D_s_c"] = ca.ds

    # INIT
    params["ce0"] = ic.ce
    params["ce_a0"] = ic.ce
    params["ce_c0"] = ic.ce
    params["cs_a0"] = ic.an.cs
    params["cs_c0"] = ic.ca.cs
    params["eps_s_a"] = an.solids.eps
    params["eps_s_c"] = ca.solids.eps
    params["L_a"] = an.thickness
    params["L_c"] = ca.thickness
    j_a = (
        -(params["I_discharge"] / params["A_a"])
        * params["Rs_a"]
        / (np.float64(3.0) * params["eps_s_a"] * params["F"] * params["L_a"])
    )  # OK
    j_c = (
        (params["I_discharge"] / params["A_c"])
        * params["Rs_c"]
        / (np.float64(3.0) * params["eps_s_c"] * params["F"] * params["L_c"])
    )
    params["j_a"] = j_a
    params["j_c"] = j_c

    cse_a = ic.an.cs
    i0_a = params["i0_a"](
        cse_a,
        params["ce0"],
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        np.float64(1.0),
    )
    Uocp_a = params["Uocp_a"](cse_a, params["csanmax"])
    params["Uocp_a0"] = Uocp_a

    params["phie0"] = phie0_fun

    cse_c = ic.ca.cs
    i0_c = params["i0_c"](
        cse_c,
        params["ce0"],
        params["T"],
        params["alpha_c"],
        params["cscamax"],
        params["R"],
    )
    params["i0_c0"] = i0_c
    Uocp_c = params["Uocp_c"](cse_c, params["cscamax"])
    params["Uocp_c0"] = Uocp_c

    params["phis_c0"] = phis_c0_fun

    # RESCALE CORRECTION
    params["rescale_cs_a"] = -ic.an.cs
    params["rescale_cs_c"] = params["cscamax"] - ic.ca.cs
    params["rescale_phis_c"] = abs(
        np.float64(3.8) - np.float64(4.110916387038547)
    )  # OK
    params["rescale_phie"] = abs(
        np.float64(-0.15) - np.float64(-0.07645356566609385)
    )  # OK

    return params
