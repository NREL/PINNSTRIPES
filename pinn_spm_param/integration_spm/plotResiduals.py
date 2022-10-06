import sys

sys.path.append("../util")
import numpy as np
from plotsUtil import *
from plotsUtil_batt import *
from spm_simpler import *

params = makeParams()

# solution
solution = np.load("solution.npz")
r_a = solution["r_a"]
r_c = solution["r_c"]
t = solution["t"]
cs_a = solution["cs_a"]
cs_c = solution["cs_c"]


# discretization
n_t = len(t)
n_r = len(r_a)
dR_a = params["Rs_a"] / (n_r - 1)
dR_c = params["Rs_c"] / (n_r - 1)
dt = params["tmax"] / (n_t - 1)
r_a_rep = np.repeat(np.reshape(r_a, (1, n_r)), n_t, axis=0)
r_c_rep = np.repeat(np.reshape(r_c, (1, n_r)), n_t, axis=0)
rhs_csa = np.zeros((n_t, n_r))
rhs_csc = np.zeros((n_t, n_r))
res_csa = np.zeros((n_t, n_r))
res_csc = np.zeros((n_t, n_r))

Ds_a = np.zeros(n_r)
Ds_c = np.zeros(n_r)
rhs_a = np.zeros(n_r)
rhs_c = np.zeros(n_r)

res_j_an = np.zeros(n_t)
res_j_ca = np.zeros(n_t)

# initialize
ce = params["ce0"]
phis_a = 0

j_a = -params["I_discharge"] / (params["A_a"] * params["F"])
j_c = params["I_discharge"] / (params["A_c"] * params["F"])

# GET cs_a
Ds_a = params["D_s_a"](params["T"], params["R"])
r2_Dsa = Ds_a * r_a_rep**2
ddr_csa = np.gradient(cs_a, r_a, axis=1, edge_order=2)
r2_Dsa_ddr_csa = r2_Dsa * ddr_csa
ddr_r2_Dsa_ddr_csa = np.gradient(r2_Dsa_ddr_csa, r_a, axis=1, edge_order=2)
rhs_csa[:, 1:] = (1 / (r_a_rep[:, 1:] ** 2)) * ddr_r2_Dsa_ddr_csa[:, 1:]
ddt_csa = np.gradient(cs_a, t, axis=0, edge_order=2)
res_csa[:, 1:] = ddt_csa[:, 1:] - rhs_csa[:, 1:]
rescale_csa = (1 / (params["Rs_a"] ** 2)) * Ds_a * params["cs_a0"]

# GET cs_c
Ds_c = params["D_s_c"](cs_c, params["T"], params["R"], params["cscamax"])
r2_Dsc = Ds_c * r_c_rep**2
ddr_csc = np.gradient(cs_c, r_c, axis=1, edge_order=2)
r2_Dsc_ddr_csc = r2_Dsc * ddr_csc
ddr_r2_Dsc_ddr_csc = np.gradient(r2_Dsc_ddr_csc, r_c, axis=1, edge_order=2)
rhs_csc[:, 1:] = (1 / (r_c_rep[:, 1:] ** 2)) * ddr_r2_Dsc_ddr_csc[:, 1:]
ddt_csc = np.gradient(cs_c, t, axis=0, edge_order=2)
res_csc[:, 1:] = ddt_csc[:, 1:] - rhs_csc[:, 1:]
rescale_csc = (1 / (params["Rs_c"] ** 2)) * np.mean(Ds_c) * params["cs_c0"]

fig = plt.figure()
plt.plot(
    r_a / params["Rs_a"],
    res_csa[0, :] / rescale_csa,
    linewidth=3,
    color="k",
    label=r"$c_{s,a}$/$c_{s,an,max}$",
)
plt.plot(
    r_c / params["Rs_c"],
    res_csc[0, :] / rescale_csc,
    "-^",
    linewidth=3,
    color="k",
    label=r"$c_{s,c}$/$c_{s,ca,max}$",
)
nLines = 10
for i in range(nLines):
    plt.plot(
        r_a / params["Rs_a"],
        res_csa[i * (n_t - 1) // 10, :] / rescale_csa,
        linewidth=3,
        color=str(i / int(nLines * 1.5)),
    )
    plt.plot(
        r_c / params["Rs_c"],
        res_csc[i * (n_t - 1) // 10, :] / rescale_csc,
        "-^",
        linewidth=3,
        color=str(i / int(nLines * 1.5)),
    )
prettyLabels("r/R", "", 14)
plotLegend()


# Plot cs
plotData(
    [r_a, r_c],
    [abs(res_csa / rescale_csa), abs(res_csc / rescale_csc)],
    params["tmax"],
    [r"", r""],
    [r"Res $c_{s,an}/c_{s,an,max}$", r"Res $c_{s,ca}/c_{s,ca,max}$"],
    ["r [m]", "r [m]"],
    vminList=[0, 0],
    vmaxList=[1e-1, 1e-1],
)

plt.show()
