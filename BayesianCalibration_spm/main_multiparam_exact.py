import os
import sys

import numpy as np

sys.path.append("../pinn_spm_param/util")
import argument
import tensorflow as tf
from myNN import *
from plotsUtil import *
from tensorflow import keras
from tensorflow.keras import layers, regularizers

tf.keras.backend.set_floatx("float64")

# Read command line arguments
args = argument.initArg()

import corner  # plotting package
import emcee  # montecarlo sampler
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib import rc, rcParams
from plotsUtil import *
from spm_simpler import *

nT_target = args.n_t

data_phis_c = np.load("dataMeasured_%d.npz" % nT_target)["data"].astype(
    "float64"
)
data_t = np.load("dataMeasured_%d.npz" % nT_target)["t"].astype("float64")
n_t = data_t.shape[0]
inpt_grid = np.zeros((n_t, 1)).astype("float64")
inpt_grid[:, 0] = data_t

params = makeParams()
params_min = [0.5, 1]
params_max = [4, 10]


def forwardModel(unknowns, inpt):
    forwardModel.counter += 1
    # discretization
    n_r = 16
    r_a = np.linspace(0, params["Rs_a"], n_r)
    dR_a = params["Rs_a"] / (n_r - 1)
    r_c = np.linspace(0, params["Rs_c"], n_r)
    dR_c = params["Rs_c"] / (n_r - 1)

    deg_i0_a = np.clip(unknowns[0], params_min[0], params_max[0])
    deg_ds_c = np.clip(unknowns[1], params_min[1], params_max[1])

    ds_c = params["D_s_c"](
        params["cs_c0"], params["T"], params["R"], params["cscamax"], deg_ds_c
    )

    mindR = min(dR_a, dR_c)

    Ds_ave = 0.5 * (params["D_s_a"](params["T"], params["R"]) + ds_c)

    dt = mindR**2 / (4 * Ds_ave)
    n_t = int(0.5 * params["tmax"] // dt)

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

        t_measure = np.reshape(inpt, (inpt.shape[0], 1)).astype("float64")
        data_measure = np.reshape(
            np.interp(t_measure, t, phis_c), (inpt.shape[0], 1)
        ).astype("float64")
    return data_measure


forwardModel.counter = 0


def lnlikelihood(unknowns, data_meas, inpt_grid):
    theta = unknowns[:2]
    # sig = unknowns[2]
    # if sig < 1e-16:
    #    sig = 1e-16
    f = forwardModel(theta, inpt_grid)
    y = np.reshape(data_meas, (n_t, 1))
    f = np.reshape(f, (n_t, 1))
    sigma = 20
    likelihood = -0.5 * np.sum((sigma * (y - f)) ** 2) + np.log(
        2 * np.pi * sigma**2
    )
    # print('like=%.2f ds_a=%.2f ds_c=%.2f' % (likelihood,unknowns[0],unknowns[1]))
    return likelihood


def lnprior(unknowns):
    theta = unknowns[:2]
    # sig = unknowns[1]
    if (
        params_min[0] < theta[0] < params_max[0]
        and params_min[1] < theta[1] < params_max[1]
    ):
        return 0.0
    return -np.inf


def lnposterior(unknowns, data_meas, inpt_grid):
    lpr = lnprior(unknowns)
    if not np.isfinite(lpr):
        return -np.inf
    llik = lnlikelihood(unknowns, data_meas, inpt_grid)
    lpos = llik + lpr
    return lpos


# Optimization of the posterior
guess = np.array([2, 2])
nlp = lambda *args: -lnlikelihood(*args)
out = opt.minimize(nlp, guess, args=(data_phis_c, data_t))

print(out)
print("")
print("theta1_opt: ", out["x"][0])
print("theta2_opt: ", out["x"][1])
# print('sigma_opt: ',out['x'][2])
# Ensure positivity of sigma
# out['x'][2] = abs(out['x'][2])

## Plot prediction
# A1 = np.reshape(forwardModel(out['x'],inpt_grid),(n_t,n_x))
# fig = plt.figure()
# for i_t in range(n_t):
#    plt.plot(data_x,data_phi[i_t,:],'o',color='k',label='data ')
#    plt.plot(data_x,A1[i_t,:],'x',color='k',label='calibrated ')
# prettyLabels('x',r'$\phi$',14)
# plt.legend()
# plt.show()


# Get distribution
pos = guess + 1e-3 * np.random.randn(16, 2)
pos[:, 0] = np.clip(pos[:, 0], params_min[0], params_max[0])
pos[:, 1] = np.clip(pos[:, 1], params_min[1], params_max[1])
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, lnposterior, args=([data_phis_c, inpt_grid])
)
sampler.run_mcmc(pos, 1500, progress=True)
fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = [r"$i0_a$", r"$Ds_c$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
# This integrated autocorrelation time gives us an estimate of the number of stepes needed to forget
# where the chain started
# tau = sampler.get_autocorr_time()
# print(tau)
# discard the first 100 samples and flaten the samples to the dimmensionality of the problem: 3
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
print(flat_samples)

np.save("result_exact_%d.npy" % nT_target, flat_samples)

import corner

# activate latex text rendering
rc("font", weight="bold", family="Times New Roman", size=14)
fig = corner.corner(
    flat_samples,
    plot_datapoints=False,
    bins=300,
    smooth=1,
    smooth1d=1,
    labels=labels,
    truths=[2, 2],
    use_math_text=True,
    label_kwargs={
        "fontsize": 14,
        "fontweight": "bold",
        "fontname": "Times New Roman",
    },
    title_kwargs={
        "fontsize": 14,
        "fontweight": "bold",
        "fontname": "Times New Roman",
    },
)


theta1_sample, theta2_sample = np.median(flat_samples, axis=0)
# Plot prediction
A_opt = forwardModel(out["x"][:2], inpt_grid)
A_median = forwardModel([theta1_sample, theta2_sample], inpt_grid)

print("Model evaluations = ", forwardModel.counter)
print("Median %.2f %.2f = " % (theta1_sample, theta2_sample))

plt.show()
