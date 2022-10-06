import sys

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx("float64")


def uocp_a_fun_mod(cs_a, csanmax):
    x = cs_a / csanmax
    return (
        np.float64(-1.059423355572770e-02)
        * tf.tanh(
            (x - np.float64(1.453708425609560e-02))
            / np.float64(9.089868397988610e-05)
        )
        + np.float64(2.443615203087110e-02)
        * tf.tanh(
            (x - np.float64(5.464261369950400e-01))
            / np.float64(6.270508166379020e-01)
        )
        + np.float64(-1.637520788053810e-02)
        * tf.tanh(
            (x - np.float64(5.639025014475490e-01))
            / np.float64(7.053886409518520e-02)
        )
        + np.float64(-6.542365622896410e-02)
        * tf.tanh(
            (x - np.float64(5.960370524233590e-01))
            / np.float64(1.409966536648620e00)
        )
        + np.float64(-4.173226059293490e-02)
        * tf.tanh(
            (x - np.float64(1.787670587868640e-01))
            / np.float64(7.693844911793470e-02)
        )
        - np.float64(4.792178163846890e-01)
        * tf.tanh(
            (x + np.float64(3.845707852011820e-03))
            / np.float64(4.112633446959460e-02)
        )
        + np.float64(6.594735004847470e-01)
        - np.float64(4.364293924074990e-02)
        * tf.tanh(
            (x - np.float64(9.449231893318330e-02))
            / np.float64(-2.046776012570780e-02)
        )
        - np.float64(8.241166396760410e-02)
        * tf.tanh(
            (x - np.float64(7.746685789572230e-02))
            / np.float64(3.593817905677970e-02)
        )
        + (
            (
                -np.float64(1.731504647676420e02) * x**8
                + np.float64(8.252008712749000e01) * x**7
                + np.float64(1.233160814852810e02) * x**6
                + np.float64(5.913206621637760e01) * x**5
                + np.float64(3.322960033709470e01) * x**4
                + np.float64(3.437968012320620e00) * x**3
                - np.float64(6.906367679257650e01) * x**2
                - np.float64(1.228217254296760e01) * x
                - np.float64(40)
            )
            - np.float64(-1.059423355572770e-02)
            * tf.tanh(
                (x - np.float64(1.453708425609560e-02))
                / np.float64(9.089868397988610e-05)
            )
            + np.float64(2.443615203087110e-02)
            * tf.tanh(
                (x - np.float64(5.464261369950400e-01))
                / np.float64(6.270508166379020e-01)
            )
            + np.float64(-1.637520788053810e-02)
            * tf.tanh(
                (x - np.float64(5.639025014475490e-01))
                / np.float64(7.053886409518520e-02)
            )
            + np.float64(-6.542365622896410e-02)
            * tf.tanh(
                (x - np.float64(5.960370524233590e-01))
                / np.float64(1.409966536648620e00)
            )
            + np.float64(-4.173226059293490e-02)
            * tf.tanh(
                (x - np.float64(1.787670587868640e-01))
                / np.float64(7.693844911793470e-02)
            )
            - np.float64(4.792178163846890e-01)
            * tf.tanh(
                (x + np.float64(3.845707852011820e-03))
                / np.float64(4.112633446959460e-02)
            )
            + np.float64(6.594735004847470e-01)
            - np.float64(4.364293924074990e-02)
            * tf.tanh(
                (x - np.float64(9.449231893318330e-02))
                / np.float64(-2.046776012570780e-02)
            )
            - np.float64(8.241166396760410e-02)
            * tf.tanh(
                (x - np.float64(7.746685789572230e-02))
                / np.float64(3.593817905677970e-02)
            )
        )
        / (
            np.float64(1.0)
            + tf.exp(-np.float64(80) * (x - np.float64(1.02956203215198)))
        )
    )


def uocp_a_simp(cs_a, csanmax):
    x = cs_a / csanmax
    return np.float64(0.2) - np.float64(0.2) * x


def uocp_a_fun(cs_a, csanmax):
    x = cs_a / csanmax
    return (
        np.float64(-1.059423355572770e-02)
        * tf.tanh(
            (x - np.float64(1.453708425609560e-02))
            / np.float64(9.089868397988610e-05)
        )
        + np.float64(2.443615203087110e-02)
        * tf.tanh(
            (x - np.float64(5.464261369950400e-01))
            / np.float64(6.270508166379020e-01)
        )
        + np.float64(-1.637520788053810e-02)
        * tf.tanh(
            (x - np.float64(5.639025014475490e-01))
            / np.float64(7.053886409518520e-02)
        )
        + np.float64(-6.542365622896410e-02)
        * tf.tanh(
            (x - np.float64(5.960370524233590e-01))
            / np.float64(1.409966536648620e00)
        )
        + np.float64(-4.173226059293490e-02)
        * tf.tanh(
            (x - np.float64(1.787670587868640e-01))
            / np.float64(7.693844911793470e-02)
        )
        - np.float64(4.792178163846890e-01)
        * tf.tanh(
            (x + np.float64(3.845707852011820e-03))
            / np.float64(4.112633446959460e-02)
        )
        + np.float64(6.594735004847470e-01)
        - np.float64(4.364293924074990e-02)
        * tf.tanh(
            (x - np.float64(9.449231893318330e-02))
            / np.float64(-2.046776012570780e-02)
        )
        - np.float64(8.241166396760410e-02)
        * tf.tanh(
            (x - np.float64(7.746685789572230e-02))
            / np.float64(3.593817905677970e-02)
        )
        + (
            (
                -np.float64(1.731504647676420e02) * x**8
                + np.float64(8.252008712749000e01) * x**7
                + np.float64(1.233160814852810e02) * x**6
                + np.float64(5.913206621637760e01) * x**5
                + np.float64(3.322960033709470e01) * x**4
                + np.float64(3.437968012320620e00) * x**3
                - np.float64(6.906367679257650e01) * x**2
                - np.float64(1.228217254296760e01) * x
                - np.float64(5.037944982759270e01)
            )
            - np.float64(-1.059423355572770e-02)
            * tf.tanh(
                (x - np.float64(1.453708425609560e-02))
                / np.float64(9.089868397988610e-05)
            )
            + np.float64(2.443615203087110e-02)
            * tf.tanh(
                (x - np.float64(5.464261369950400e-01))
                / np.float64(6.270508166379020e-01)
            )
            + np.float64(-1.637520788053810e-02)
            * tf.tanh(
                (x - np.float64(5.639025014475490e-01))
                / np.float64(7.053886409518520e-02)
            )
            + np.float64(-6.542365622896410e-02)
            * tf.tanh(
                (x - np.float64(5.960370524233590e-01))
                / np.float64(1.409966536648620e00)
            )
            + np.float64(-4.173226059293490e-02)
            * tf.tanh(
                (x - np.float64(1.787670587868640e-01))
                / np.float64(7.693844911793470e-02)
            )
            - np.float64(4.792178163846890e-01)
            * tf.tanh(
                (x + np.float64(3.845707852011820e-03))
                / np.float64(4.112633446959460e-02)
            )
            + np.float64(6.594735004847470e-01)
            - np.float64(4.364293924074990e-02)
            * tf.tanh(
                (x - np.float64(9.449231893318330e-02))
                / np.float64(-2.046776012570780e-02)
            )
            - np.float64(8.241166396760410e-02)
            * tf.tanh(
                (x - np.float64(7.746685789572230e-02))
                / np.float64(3.593817905677970e-02)
            )
        )
        / (
            np.float64(1.0)
            + tf.exp(-np.float64(1.0e2) * (x - np.float64(1.02956203215198)))
        )
    )


def uocp_c_fun(cs_c, cscamax):
    x = cs_c / cscamax
    return (
        np.float64(5.314735633000300e00)
        - np.float64(3.640117692001490e03) * x**14
        + np.float64(1.317657544484270e04) * x**13
        - np.float64(1.455742062291360e04) * x**12
        - np.float64(1.571094264365090e03) * x**11
        + np.float64(1.265630978512400e04) * x**10
        - np.float64(2.057808873526350e03) * x**9
        - np.float64(1.074374333186190e04) * x**8
        + np.float64(8.698112755348720e03) * x**7
        - np.float64(8.297904604107030e02) * x**6
        - np.float64(2.073765547574810e03) * x**5
        + np.float64(1.190223421193310e03) * x**4
        - np.float64(2.724851668445780e02) * x**3
        + np.float64(2.723409218042130e01) * x**2
        - np.float64(4.158276603609060e00) * x
        - np.float64(5.573191762723310e-04)
        * tf.exp(
            np.float64(6.560240842659690e00)
            * tf.math.maximum(x, np.float64(0.0))
            ** np.float64(4.148209275061330e01)
        )
    )


def uocp_c_simp(cs_c, cscamax):
    x = cs_c / cscamax
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


# def i0_c_simp(cs_c_max,ce,T,alpha,cscamax,R):
#    return np.float64(0.4)*np.ones(ce.shape,dtype='float64')


def i0_c_simp(cs_c_max, ce, T, alpha, cscamax, R):
    return np.float64(3.0) * np.ones(ce.shape, dtype="float64")


# def ds_a_fun(T,R):
#    return np.float64(3.0e-14) * tf.exp((np.float64(-30.0e6)/R)*(np.float64(1.0)/T - np.float64(1.0/303.15)))


def ds_a_fun_simp(T, R):
    return np.float64(3.0e-14)
    # return np.float64(3.0e-13)


# def ds_c_fun(cs_c,T,R,cscamax):
#    x = cs_c/cscamax
#    power = (-np.float64(2.509010843479270E+02)*x**10
#           + np.float64(2.391026725259970E+03)*x**9
#           - np.float64(4.868420267611360E+03)*x**8
#           - np.float64(8.331104102921070E+01)*x**7
#           + np.float64(1.057636028329000E+04)*x**6
#           - np.float64(1.268324548348120E+04)*x**5
#           + np.float64(5.016272167775530E+03)*x**4
#           + np.float64(9.824896659649480E+02)*x**3
#           - np.float64(1.502439339070900E+03)*x**2
#           + np.float64(4.723709304247700E+02)*x
#           - np.float64(6.526092046397090E+01))
#    return (np.float64(1.5)* ( np.float64(1.5)*np.float64(10.0)**(power) )
#              * tf.exp((np.float64(-30.0e6)/R) * (np.float64(1.0)/T - np.float64(1.0/303.15))))


def ds_c_fun_simp(cs_c, T, R, cscamax):
    # return np.float64(3.5e-15)*np.ones(cs_c.shape,dtype='float64')
    # return np.float64(3.0e-14)*np.ones(cs_c.shape,dtype='float64')
    return np.float64(3.5e-15) * np.ones(cs_c.shape, dtype="float64")


def ds_c_fun_plot(cs_c, T, R, cscamax):
    x = cs_c / cscamax
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


# def de_fun(ce,T):
#    power = -np.float64(0.5688226) - np.float64(1607.003)/ (T - (-np.float64(24.83763) + np.float64(64.07366) * ce)) + (-np.float64(0.8108721) + np.float64(475.291)/ (T - (-np.float64(24.83763) + np.float64(64.07366) * ce)))*ce + (-np.float64(0.005192312) - np.float64(33.43827)/ (T - (-np.float64(24.83763) + np.float64(64.07366) * ce)))*ce*ce
#    return np.float64(0.0001) * tf.math.pow(np.float64(10.0), power)


def de_fun_simp(ce, T):
    return np.float64(1.8e-10) * np.ones(ce.shape, dtype="float64")


# def ke_fun(ce,T):
#    return ce*( (np.float64(0.0001909446)*T**2-np.float64(0.08038545)*T+np.float64(9.00341))
#               +ce*(-np.float64(0.00000002887587)*T**4 + np.float64(0.00003483638)*T**3 - np.float64(0.01583677)*T**2 + np.float64(3.195295)*T - np.float64(241.4638))
#               +ce**2*(np.float64(0.00000001653786)*T**4 - np.float64(0.0000199876)*T**3 + np.float64(0.009071155)*T**2 - np.float64(1.828064)*T + np.float64(138.0976))
#               +ce**3*(-np.float64(0.000000002791965)*T**4 + np.float64(0.000003377143)*T**3 - np.float64(0.001532707)*T**2 + np.float64(0.3090003)*T - np.float64(23.35671))
#              )


def ke_fun_simp(ce, T):
    return np.float64(0.6) * np.ones(ce.shape, dtype="float64")


# def dlnf_dce_fun(ce,T):
#    return np.float64(0.54)*ce**2 * tf.exp(np.float64(329.0)/T) - np.float64(0.00225)*ce*tf.exp(np.float64(1360.0)/T) + np.float64(0.341)*tf.exp(np.float64(261.0)/T)


def dlnf_dce_fun_simp(ce, T):
    return np.float64(8.0) * np.ones(ce.shape, dtype="float64")


# def t0_fun(ce,T):
#    return (    (np.float64(-0.0000002876102)*T**2 + np.float64(0.0002077407)*T - np.float64(0.03881203))*ce**2
#              + (np.float64(0.000001161463)*T**2 - np.float64(0.00086825)*T + np.float64(0.1777266))*ce
#              + (-np.float64(0.0000006766258)*T**2 + np.float64(0.0006389189)*T + np.float64(0.3091761))
#           )


def t0_fun_simp(ce, T):
    return np.float64(0.46) * np.ones(ce.shape, dtype="float64")


def phie0_fun(i0_a, j_a, F, R, T, Uocp_a0):
    return -j_a * (F / i0_a) * (R * T / F) - Uocp_a0


def phis_c0_fun(i0_a, j_a, F, R, T, Uocp_a0, j_c, i0_c, Uocp_c0):
    phie0 = phie0_fun(i0_a, j_a, F, R, T, Uocp_a0)
    return j_c * (F / i0_c) * (R * T / F) + Uocp_c0 + phie0


def makeParams():

    params = {}

    # Parametric domain
    params["deg_i0_a_min"] = np.float64(0.5)
    params["deg_i0_a_max"] = np.float64(4)
    params["deg_ds_c_min"] = np.float64(1.0)
    params["deg_ds_c_max"] = np.float64(10)

    params["param_eff"] = np.float64(1.0)
    params["deg_i0_a_ref"] = np.float64(1.0)
    params["deg_ds_c_ref"] = np.float64(1.0)
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
    params["tmin"] = np.float64(0)
    params["tmax"] = np.float64(200)
    params["rmin"] = np.float64(0)

    # Params fixed
    params["A_a"] = np.float64(1.4e-3)  # OK
    params["A_c"] = np.float64(1.4e-3)  # OK
    params["F"] = np.float64(9.6485e7)  # OK
    params["R"] = np.float64(8.3145e3)  # OK
    params["T"] = np.float64(305.15)  # OK
    params["C"] = np.float64(-2.0)  # OK
    params["I_discharge"] = np.float64(1.89e-2 * params["C"])  # OK

    params["alpha_a"] = np.float64(0.5)  # OK
    params["alpha_c"] = np.float64(0.5)  # OK

    # Params to fit
    params["Rs_a"] = np.float64(4e-6)  # OK
    params["Rs_c"] = np.float64(1.8e-6)  # OK
    params["rescale_R"] = np.float64(max(4e-6, 1.8e-6))  # OK
    # current
    params["csanmax"] = np.float64(30)  # OK
    params["cscamax"] = np.float64(49.6)  # OK
    params["rescale_T"] = params["tmax"]  # OK

    # Typical variables magnitudes
    params["mag_cs_a"] = np.float64(25)  # OK
    params["mag_cs_c"] = np.float64(32.5)  # OK
    params["mag_phis_c"] = np.float64(4.25)  # OK
    params["mag_phie"] = np.float64(0.15)  # OK
    params["mag_ce"] = np.float64(1.2)  # OK

    # FUNCTIONS
    params["Uocp_a"] = uocp_a_simp  # OK
    params["Uocp_c"] = uocp_c_simp  # OK
    params["i0_a"] = i0_a_simp_degradation_param  # OK
    params["i0_c"] = i0_c_simp  # OK
    params["D_s_a"] = ds_a_fun_simp  # OK
    params["D_s_c"] = ds_c_fun_simp_degradation_param  # OK

    # INIT
    params["ce0"] = np.float64(1.2)  # OK
    params["cs_a0"] = np.float64(
        0.91 * params["csanmax"]
    )  # OK #CHECK THIS BEFORE CAL
    params["cs_c0"] = np.float64(
        0.39 * params["cscamax"]
    )  # OK #CHECK THIS BEFORE CAL
    # params["phis_a0"] = np.float64(0.0)
    # params["phis_c0"] = params["Uocp_c"](
    #    params["cs_c0"], params["cscamax"]
    # ) - params["Uocp_a"](params["cs_a0"], params["csanmax"])
    # params["phie0"] = -params["Uocp_a"](params["cs_a0"], params["csanmax"])
    # Get init for phie and phis
    # j_a = -params["I_discharge"] / (params["A_a"] * params["F"])
    # j_c = params["I_discharge"] / (params["A_c"] * params["F"])
    params["eps_s_a"] = np.float64(0.5430727763)
    params["eps_s_c"] = np.float64(0.47662)
    params["L_a"] = np.float64(4.4 * 1e-5)
    params["L_c"] = np.float64(4.2e-5)
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

    cse_a = params["cs_a0"]
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

    cse_c = params["cs_c0"]
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
    params["rescale_phie"] = abs(np.float64(-0.055) - np.float64(-0.04))  # OK
    params["rescale_cs_a"] = abs(np.float64(25) - params["cs_a0"])  # OK
    params["rescale_cs_c"] = abs(np.float64(22) - params["cs_c0"])  # OK
    params["rescale_phis_c"] = abs(np.float64(4.3) - np.float64(4.5))  # OK

    return params


if __name__ == "__main__":
    from plotsUtil import *

    params = makeParams()

    # UOCP_a
    fig = plt.figure()
    cs = np.linspace(0, params["csanmax"], 100)
    u = params["Uocp_a"](cs, params["csanmax"])
    plt.plot(cs, u, color="k", linewidth=3)
    prettyLabels("cs", "U [V]", 14, r"U$_{ocp,an}$")
    # um = uocp_a_fun_mod(cs,np.float64(30))
    # plt.plot(cs,u,'o',markerfacecolor='none')
    # plt.plot(cs,um,'x')

    # UOCP_c
    fig = plt.figure()
    cs = np.linspace(0, params["cscamax"], 100)
    u = params["Uocp_c"](cs, params["cscamax"])
    plt.plot(cs, u, color="k", linewidth=3)
    prettyLabels("cs", "U [V]", 14, r"U$_{ocp,ca}$")

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
        np.float64(1.0),
    )
    plt.plot(ce, np.array(i0), color="k", linewidth=3)
    prettyLabels("ce", "i", 14, r"I$_{0,a}$, cmax=csanmax/2")

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
    prettyLabels("ce", "i", 14, r"I$_{0,c}$, cmax=csanmax/2")

    # DS_a
    fig = plt.figure()
    ds_a = params["D_s_a"](params["T"], params["R"])
    plt.plot(np.ones(100) * ds_a, color="k", linewidth=3)
    prettyLabels("", "Ds", 14, r"D$_{s,a}$")

    # DS_C
    fig = plt.figure()
    cs = np.linspace(0, params["cscamax"], 100)
    params["D_s_c"] = ds_c_fun_plot_simp
    ds = params["D_s_c"](cs, params["T"], params["R"], params["cscamax"])
    plt.plot(cs, ds, color="k", linewidth=3)
    prettyLabels("cs", "Ds", 14, r"D$_{s,c}$")

    plt.show()
