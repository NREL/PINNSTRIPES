import sys

sys.path.append("../util")
import argument
import numpy as np
from plotsUtil import *
from plotsUtil_batt import *
from thermo import grad_ds_a_cs_a, grad_ds_c_cs_c

# Read command line arguments
args = argument.initArg()

if args.simpleModel:
    from spm_simpler import *
else:
    from spm import *

if not args.verbose:
    import matplotlib

    matplotlib.use("Agg")

params = makeParams()

input_params = False
if not args.params_list is None:
    input_params = True
if input_params:
    deg_i0_a = np.float64(args.params_list[0])
    deg_ds_c = np.float64(args.params_list[1])
else:
    deg_i0_a = np.float64(0.5)
    deg_ds_c = np.float64(1.0)

# discretization
from spm_int import *

if args.lean:
    n_r = 32
    dt = 5
    config, sol = exec_impl(n_r, dt, params, deg_i0_a, deg_ds_c, verbose=True)

else:
    if args.explicit:
        n_r = 32

        config, sol = exec_expl(n_r, params, deg_i0_a, deg_ds_c, verbose=True)
    elif args.implicit:
        n_r = 64
        dt = 0.5
        config, sol = exec_impl(
            n_r, dt, params, deg_i0_a, deg_ds_c, verbose=True
        )


if input_params:
    np.savez(
        "solution_%.2g_%.2g.npz" % (deg_i0_a, deg_ds_c),
        t=config["t"],
        r_a=config["r_a"],
        r_c=config["r_c"],
        cs_a=sol["cs_a"],
        cs_c=sol["cs_c"],
        phie=sol["phie"],
        phis_c=sol["phis_c"],
    )
else:
    np.savez(
        "solution.npz",
        t=config["t"],
        r_a=config["r_a"],
        r_c=config["r_c"],
        cs_a=sol["cs_a"],
        cs_c=sol["cs_c"],
        phie=sol["phie"],
        phis_c=sol["phis_c"],
    )


fig = plt.figure()
plt.plot(config["t"], sol["phie"], label=r"$\phi_e$")
plt.plot(config["t"], sol["phis_c"], label=r"$\phi_{s,c}$")
plt.plot(
    config["t"],
    sol["cs_a"][:, -1] / params["csanmax"],
    label=r"$c_{surf,s,a}$/$c_{s,an,max}$",
)
plt.plot(
    config["t"],
    sol["cs_c"][:, -1] / params["cscamax"],
    label=r"$c_{surf,s,c}$/$c_{s,ca,max}$",
)
prettyLabels("t", "", 14)
plotLegend()

fig = plt.figure()
plt.plot(
    config["r_a"] / params["Rs_a"],
    sol["cs_a"][0, :] / params["csanmax"],
    linewidth=3,
    color="k",
    label=r"$c_{s,a}$/$c_{s,an,max}$",
)
plt.plot(
    config["r_c"] / params["Rs_c"],
    sol["cs_c"][0, :] / params["cscamax"],
    "-^",
    linewidth=3,
    color="k",
    label=r"$c_{s,c}$/$c_{s,ca,max}$",
)
nLines = 10
for i in range(nLines):
    plt.plot(
        config["r_a"] / params["Rs_a"],
        sol["cs_a"][i * (config["n_t"] - 1) // 10, :] / params["csanmax"],
        linewidth=3,
        color=str(i / int(nLines * 1.5)),
    )
    plt.plot(
        config["r_c"] / params["Rs_c"],
        sol["cs_c"][i * (config["n_t"] - 1) // 10, :] / params["cscamax"],
        "-^",
        linewidth=3,
        color=str(i / int(nLines * 1.5)),
    )
prettyLabels("r/R", "", 14)
plotLegend()

fig = plt.figure()
plt.plot(config["t"], sol["phie"], label=r"$\phi_e$")
plt.plot(config["t"], sol["phis_c"], label=r"$\phi_{s,c}$")
prettyLabels("t", "", 14)
plotLegend()


# Plot cs
plotData(
    [config["r_a"], config["r_c"]],
    [sol["cs_a"] / params["csanmax"], sol["cs_c"] / params["cscamax"]],
    params["tmax"],
    [r"", r""],
    [r"$c_{s,an}/c_{s,an,max}$", r"$c_{s,ca}/c_{s,ca,max}$"],
    ["r [m]", "r [m]"],
    vminList=[0, 0],
    vmaxList=[1, 1],
)

# Plot cs
plotData(
    [config["r_a"], config["r_c"]],
    [sol["cs_a"], sol["cs_c"]],
    params["tmax"],
    [r"", r""],
    [r"$c_{s,an}$", r"$c_{s,ca}$"],
    ["r [m]", "r [m]"],
)

if args.verbose:
    plt.show()
