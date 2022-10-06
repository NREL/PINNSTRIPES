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
modelFolder = args.modelFolder
print("INFO: Using modelFolder : ", modelFolder)

if not os.path.exists(os.path.join(modelFolder, "config.npy")):
    print("Looking for file ", os.path.join(modelFolder, "config.npy"))
    sys.exit("ERROR: Config file could not be found necessary")
else:
    print("INFO: Loading from config file")
    configDict = np.load(
        os.path.join(modelFolder, "config.npy"), allow_pickle="TRUE"
    ).item()
    hidden_units_t = configDict["hidden_units_t"]
    hidden_units_t_r = configDict["hidden_units_t_r"]
    hidden_units_phie = configDict["hidden_units_phie"]
    hidden_units_phis_c = configDict["hidden_units_phis_c"]
    hidden_units_cs_a = configDict["hidden_units_cs_a"]
    hidden_units_cs_c = configDict["hidden_units_cs_c"]
    try:
        n_hidden_res_blocks = configDict["n_hidden_res_blocks"]
    except:
        n_hidden_res_blocks = 0
    if n_hidden_res_blocks > 0:
        n_res_block_layers = configDict["n_res_block_layers"]
        n_res_block_units = configDict["n_res_block_units"]
    else:
        n_res_block_layers = 1
        n_res_block_units = 1
    HARD_IC_TIMESCALE = configDict["hard_IC_timescale"]
    EXP_LIMITER = configDict["exponentialLimiter"]
    ACTIVATION = configDict["activation"]
    try:
        LINEARIZE_J = configDict["linearizeJ"]
    except:
        LINEARIZE_J = True
    DYNAMIC_ATTENTION = configDict["dynamicAttentionWeights"]
    activeInt = configDict["activeInt"]
    activeBound = configDict["activeBound"]
    activeData = configDict["activeData"]
    activeReg = configDict["activeReg"]
    try:
        params_min = configDict["params_min"]
    except:
        params_min = [
            params["deg_i0_a_min"],
            params["deg_ds_c_min"],
        ]
    try:
        params_max = configDict["params_max"]
    except:
        params_max = [
            params["deg_i0_a_max"],
            params["deg_ds_c_max"],
        ]

nn = myNN(
    params=params,
    hidden_units_phie=hidden_units_phie,
    hidden_units_phis_c=hidden_units_phis_c,
    hidden_units_cs_a=hidden_units_cs_a,
    hidden_units_cs_c=hidden_units_cs_c,
    hidden_units_t=hidden_units_t,
    hidden_units_t_r=hidden_units_t_r,
    n_hidden_res_blocks=n_hidden_res_blocks,
    n_res_block_layers=n_res_block_layers,
    n_res_block_units=n_res_block_units,
    hard_IC_timescale=HARD_IC_TIMESCALE,
    exponentialLimiter=EXP_LIMITER,
    activation=ACTIVATION,
    linearizeJ=LINEARIZE_J,
    params_min=params_min,
    params_max=params_max,
)


model = nn.model
model.load_weights(os.path.join(modelFolder, "best.h5"))
# rescale
resc_r = params["rescale_R"]
resc_t = params["rescale_T"]

dummyR = np.zeros((data_t.shape[0], 1))


def forwardModel(unknowns, inpt):
    forwardModel.counter += 1
    t = np.reshape(inpt, (inpt.shape[0], 1))
    deg_i0_a = np.clip(
        unknowns[0] * np.ones((len(t), 1)).astype("float64"),
        nn.params_min[0],
        nn.params_max[0],
    )
    deg_ds_c = np.clip(
        unknowns[1] * np.ones((len(t), 1)).astype("float64"),
        nn.params_min[1],
        nn.params_max[1],
    )
    out = nn.model(
        [
            t / resc_t,
            dummyR,
            deg_i0_a / nn.resc_params[nn.ind_deg_i0_a],
            deg_ds_c / nn.resc_params[nn.ind_deg_ds_c],
        ]
    )
    cse_a = nn.rescaleCs_a(out[nn.ind_cs_a], t)
    ce = params["ce0"] * np.ones(cse_a.shape)

    i0_a = params["i0_a"](
        cse_a,
        ce,
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        deg_i0_a,
    )
    phis_c_nonrescaled = out[nn.ind_phis_c]
    phis_c_rescaled = np.reshape(
        nn.rescalePhis_c(phis_c_nonrescaled, t, i0_a), (n_t, 1)
    ).astype("float64")
    return phis_c_rescaled


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

    # if self.kmin < theta[0] < self.kmax and self.Rmin < theta[1] < self.Rmax and 0 < sig < 5:
    if (
        nn.params_min[0] < theta[0] < nn.params_max[0]
        and nn.params_min[1] < theta[1] < nn.params_max[1]
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
pos[:, 0] = np.clip(pos[:, 0], nn.params_min[0], nn.params_max[0])
pos[:, 1] = np.clip(pos[:, 1], nn.params_min[1], nn.params_max[1])
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

np.save("result_dataOnly_%d.npy" % nT_target, flat_samples)

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
