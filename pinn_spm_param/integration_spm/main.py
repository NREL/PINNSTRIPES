import sys

sys.path.append("../util")
import argument
import numpy as np
from plotsUtil import *
from plotsUtil_batt import *
from spm_simpler import *

# Read command line arguments
args = argument.initArg()

if not args.verbose:
    import matplotlib

    matplotlib.use("Agg")

params = makeParams()

input_params = False
if not args.params_list is None:
    input_params = True
if input_params:
    deg_i0_a = float(args.params_list[0])
    deg_ds_c = float(args.params_list[1])
else:
    deg_i0_a = 1.0
    deg_ds_c = 1.0

# discretization
n_r = 32
r_a = np.linspace(0, params["Rs_a"], n_r)
dR_a = params["Rs_a"] / (n_r - 1)
r_c = np.linspace(0, params["Rs_c"], n_r)
dR_c = params["Rs_c"] / (n_r - 1)

mindR = min(dR_a, dR_c)
Ds_ave = 0.5 * (params["D_s_a"](params["T"], params["R"])) + 0.5 * (
    params["D_s_c"](
        params["cs_c0"], params["T"], params["R"], params["cscamax"], deg_ds_c
    )
)

dt = mindR**2 / (2 * Ds_ave)
n_t = int(2 * params["tmax"] // dt)

t = np.linspace(0, params["tmax"], n_t)
dt = params["tmax"] / (n_t - 1)
phie = np.zeros(n_t)
phis_c = np.zeros(n_t)
cs_a = np.zeros((n_t, n_r))
cs_c = np.zeros((n_t, n_r))
Ds_a = np.zeros(n_r)
Ds_c = np.zeros(n_r)
rhs_a = np.zeros(n_r)
rhs_c = np.zeros(n_r)

# initialize
ce = params["ce0"]
phis_a = 0
cs_a[0, :] = params["cs_a0"]
cs_c[0, :] = params["cs_c0"]
# j_a = -params["I_discharge"] / (params["A_a"] * params["F"])
# j_c = params["I_discharge"] / (params["A_c"] * params["F"])
j_a = params["j_a"]
j_c = params["j_c"]

Uocp_a = params["Uocp_a"](params["cs_a0"], params["csanmax"])
i0_a = params["i0_a"](
    params["cs_a0"],
    ce,
    params["T"],
    params["alpha_a"],
    params["csanmax"],
    params["R"],
    deg_i0_a,
)
phie[0] = params["phie0"](
    i0_a,
    j_a,
    params["F"],
    params["R"],
    params["T"],
    Uocp_a,
)
Uocp_c = params["Uocp_c"](params["cs_c0"], params["cscamax"])
i0_c = params["i0_c"](
    params["cs_c0"],
    params["ce0"],
    params["T"],
    params["alpha_c"],
    params["cscamax"],
    params["R"],
)
phis_c[0] = params["phis_c0"](
    i0_a,
    j_a,
    params["F"],
    params["R"],
    params["T"],
    Uocp_a,
    j_c,
    i0_c,
    Uocp_c,
)

LINEARIZE_J = True


for i_t in range(1, n_t):
    # for i_t in range(1,2):
    # GET PHIE: -I/A = ja = (i0/F) * sinh ( (F/RT) (-phie - Uocp_a))
    cse_a = cs_a[i_t - 1, -1]
    i0_a = params["i0_a"](
        cse_a,
        ce,
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        deg_i0_a,
    )
    Uocp_a = params["Uocp_a"](cse_a, params["csanmax"])
    if not LINEARIZE_J:
        phie[i_t] = (
            -(params["R"] * params["T"] / params["F"])
            * np.arcsinh(j_a * params["F"] / i0_a)
            - Uocp_a
        )
    else:
        phie[i_t] = (
            -j_a
            * (params["F"] / i0_a)
            * (params["R"] * params["T"] / params["F"])
            - Uocp_a
        )

    # GET PHIS_c: I/A = jc = (i0/F) * sinh ( (F/RT) (phis_c -phie - Uocp_c))
    cse_c = cs_c[i_t - 1, -1]
    i0_c = params["i0_c"](
        cse_c,
        ce,
        params["T"],
        params["alpha_c"],
        params["cscamax"],
        params["R"],
    )
    Uocp_c = params["Uocp_c"](cse_c, params["cscamax"])
    if not LINEARIZE_J:
        phis_c[i_t] = (
            (params["R"] * params["T"] / params["F"])
            * np.arcsinh(j_c * params["F"] / i0_c)
            + Uocp_c
            + phie[i_t]
        )
    else:
        phis_c[i_t] = (
            j_c
            * (params["F"] / i0_c)
            * (params["R"] * params["T"] / params["F"])
            + Uocp_c
            + phie[i_t]
        )

    # GET cs_a
    Ds_a[:] = params["D_s_a"](params["T"], params["R"])
    ddr_csa = np.gradient(cs_a[i_t - 1, :], r_a, axis=0, edge_order=2)
    ddr_csa[0] = 0
    ddr_csa[-1] = -j_a / Ds_a[-1]
    ddr2_csa = np.zeros(n_r)
    ddr2_csa[1 : n_r - 1] = (
        cs_a[i_t - 1, : n_r - 2]
        - 2 * cs_a[i_t - 1, 1 : n_r - 1]
        + cs_a[i_t - 1, 2:n_r]
    ) / dR_a**2
    ddr2_csa[0] = (
        cs_a[i_t - 1, 0] - 2 * cs_a[i_t - 1, 0] + cs_a[i_t - 1, 1]
    ) / dR_a**2
    ddr2_csa[-1] = (
        cs_a[i_t - 1, -2]
        - 2 * cs_a[i_t - 1, -1]
        + cs_a[i_t - 1, -1]
        + ddr_csa[-1] * dR_a
    ) / dR_a**2
    ddr_Ds = np.gradient(Ds_a, r_a, axis=0, edge_order=2)
    rhs_a[1:] = (
        Ds_a[1:] * ddr2_csa[1:]
        + ddr_Ds[1:] * ddr_csa[1:]
        + 2 * Ds_a[1:] * ddr_csa[1:] / r_a[1:]
    )
    rhs_a[0] = 3 * Ds_a[0] * ddr2_csa[0]
    cs_a[i_t, :] = np.clip(
        cs_a[i_t - 1, :] + dt * rhs_a, a_min=0.0, a_max=None
    )

    # GET cs_c
    Ds_c[:] = params["D_s_c"](
        cs_c[i_t - 1, :],
        params["T"],
        params["R"],
        params["cscamax"],
        deg_ds_c * np.ones(cs_c[i_t - 1, :].shape),
    )
    ddr_csc = np.gradient(cs_c[i_t - 1, :], r_c, axis=0, edge_order=2)
    ddr_csc[0] = 0
    ddr_csc[-1] = -j_c / Ds_c[-1]
    ddr2_csc = np.zeros(n_r)
    ddr2_csc[1 : n_r - 1] = (
        cs_c[i_t - 1, : n_r - 2]
        - 2 * cs_c[i_t - 1, 1 : n_r - 1]
        + cs_c[i_t - 1, 2:n_r]
    ) / dR_c**2
    ddr2_csc[0] = (
        cs_c[i_t - 1, 0] - 2 * cs_c[i_t - 1, 0] + cs_c[i_t - 1, 1]
    ) / dR_c**2
    ddr2_csc[-1] = (
        cs_c[i_t - 1, -2]
        - 2 * cs_c[i_t - 1, -1]
        + cs_c[i_t - 1, -1]
        + ddr_csc[-1] * dR_c
    ) / dR_c**2
    ddr_Ds = np.gradient(Ds_c, r_c, axis=0, edge_order=2)
    rhs_c[1:] = (
        Ds_c[1:] * ddr2_csc[1:]
        + ddr_Ds[1:] * ddr_csc[1:]
        + 2 * Ds_c[1:] * ddr_csc[1:] / r_c[1:]
    )
    rhs_c[0] = 3 * Ds_c[0] * ddr2_csc[0]
    cs_c[i_t, :] = np.clip(
        cs_c[i_t - 1, :] + dt * rhs_c, a_min=0.0, a_max=None
    )

if input_params:
    np.savez(
        "solution_%.2g_%.2g.npz" % (deg_i0_a, deg_ds_c),
        t=t,
        r_a=r_a,
        r_c=r_c,
        cs_a=cs_a,
        cs_c=cs_c,
        phie=phie,
        phis_c=phis_c,
    )
else:
    np.savez(
        "solution.npz",
        t=t,
        r_a=r_a,
        r_c=r_c,
        cs_a=cs_a,
        cs_c=cs_c,
        phie=phie,
        phis_c=phis_c,
    )


fig = plt.figure()
plt.plot(t, phie, label=r"$\phi_e$")
plt.plot(t, phis_c, label=r"$\phi_{s,c}$")
plt.plot(
    t, cs_a[:, -1] / params["csanmax"], label=r"$c_{surf,s,a}$/$c_{s,an,max}$"
)
plt.plot(
    t, cs_c[:, -1] / params["cscamax"], label=r"$c_{surf,s,c}$/$c_{s,ca,max}$"
)
prettyLabels("t", "", 14)
plotLegend()

fig = plt.figure()
plt.plot(
    r_a / params["Rs_a"],
    cs_a[0, :] / params["csanmax"],
    linewidth=3,
    color="k",
    label=r"$c_{s,a}$/$c_{s,an,max}$",
)
plt.plot(
    r_c / params["Rs_c"],
    cs_c[0, :] / params["cscamax"],
    "-^",
    linewidth=3,
    color="k",
    label=r"$c_{s,c}$/$c_{s,ca,max}$",
)
nLines = 10
for i in range(nLines):
    plt.plot(
        r_a / params["Rs_a"],
        cs_a[i * (n_t - 1) // 10, :] / params["csanmax"],
        linewidth=3,
        color=str(i / int(nLines * 1.5)),
    )
    plt.plot(
        r_c / params["Rs_c"],
        cs_c[i * (n_t - 1) // 10, :] / params["cscamax"],
        "-^",
        linewidth=3,
        color=str(i / int(nLines * 1.5)),
    )
prettyLabels("r/R", "", 14)
plotLegend()

fig = plt.figure()
plt.plot(t, phie, label=r"$\phi_e$")
plt.plot(t, phis_c, label=r"$\phi_{s,c}$")
prettyLabels("t", "", 14)
plotLegend()


# Plot cs
plotData(
    [r_a, r_c],
    [cs_a / params["csanmax"], cs_c / params["cscamax"]],
    params["tmax"],
    [r"", r""],
    [r"$c_{s,an}/c_{s,an,max}$", r"$c_{s,ca}/c_{s,ca,max}$"],
    ["r [m]", "r [m]"],
    vminList=[0, 0],
    vmaxList=[1, 1],
)

# Plot cs
plotData(
    [r_a, r_c],
    [cs_a, cs_c],
    params["tmax"],
    [r"", r""],
    [r"$c_{s,an}$", r"$c_{s,ca}$"],
    ["r [m]", "r [m]"],
)

if args.verbose:
    plt.show()
