import argparse
import json
import os
import sys

import keras
import numpy as np
import tensorflow as tf
import tf2jax
from keras import layers

parser = argparse.ArgumentParser(description="BNN interface for pouch cells")
parser.add_argument(
    "-uf",
    "--utilFolder",
    type=str,
    metavar="",
    required=True,
    help="util folder of model",
    default=None,
)
parser.add_argument(
    "-n_t",
    "--n_t",
    type=int,
    metavar="",
    required=False,
    help="number of measurements",
    default=100,
)
parser.add_argument(
    "-noise",
    "--noise",
    type=float,
    metavar="",
    required=False,
    help="noise level",
    default=0,
)
parser.add_argument(
    "-minsigma",
    "--minsigma",
    type=float,
    metavar="",
    required=False,
    help="min val for sigma",
    default=1e-2,
)

args, unknown = parser.parse_known_args()
min_sigma = args.minsigma
n_t = args.n_t
noise = args.noise

sys.path.append(args.utilFolder)
import argument
import corner
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from init_pinn import initialize_nn_from_params_config, safe_load
from numpyro.infer import MCMC, NUTS, SA
from plotsUtil import plotLegend, prettyLabels

# Read command line arguments
args_spm = argument.initArg()

if args_spm.simpleModel:
    from spm_simpler import *
else:
    from spm import *
params = makeParams()
params["deg_params_names"] = ["i0_a", "ds_c"]
params["n_params"] = 2

modelFolder = args_spm.modelFolder
print("INFO: Loading from config file")
with open(os.path.join(modelFolder, "config.json")) as json_file:
    configDict = json.load(json_file)
nn = initialize_nn_from_params_config(params, configDict)
nn = safe_load(nn, os.path.join(modelFolder, "best.h5"))
params_min = configDict["params_min"]
params_max = configDict["params_max"]


def load_observation_data(n_t=n_t, noise=noise):
    filename = f"dataMeasured_{n_t}_{noise:.2g}.npz"
    print("loading = ", filename)
    data_phis_c = np.load(filename)["data"].astype("float32")
    deg_param_truth = [2, 2]
    data_t = np.load(filename)["t"].astype("float32")
    return data_t, data_phis_c, deg_param_truth


data_t, data_phis_c, deg_param_truth = load_observation_data()
dummyR = params["Rs_c"] * np.ones((data_t.shape[0], 1))
x = (params["L_a"] + params["L_s"] + params["L_c"]) * np.ones(
    (data_t.shape[0], 1)
)
t = np.reshape(data_t, (data_t.shape[0], 1))
t_tens = tf.convert_to_tensor(t, dtype=tf.dtypes.float64)
x_tens = tf.convert_to_tensor(x, dtype=tf.dtypes.float64)
dummyR_tens = (
    tf.convert_to_tensor(dummyR, dtype=tf.dtypes.float64)
    / nn.params["rescale_R"]
)
t_tens_resc = tf.convert_to_tensor(t) / nn.params["rescale_T"]
x_tens_resc = tf.convert_to_tensor(x) / nn.params["rescale_x"]
ones_tf64 = tf.ones(tf.shape(t_tens), dtype=tf.dtypes.float64)


@tf.function
def forward(p):
    deg_par = [p[i] * ones_tf64 for i in range(len(p))]
    deg_par_resc = [
        nn.rescale_param(p[i], i) * ones_tf64 for i in range(len(p))
    ]
    out = nn.model([t_tens_resc, x_tens_resc, dummyR_tens] + deg_par_resc)
    phis_c_unrescaled = out[nn.ind_phis_c]
    phis_c = nn.rescalePhis_c(phis_c_unrescaled, t_tens, x_tens, *deg_par)[
        :, 0
    ]
    return phis_c


size_inpt = params["n_params"]
p = np.random.normal(size=(size_inpt,)).astype(np.float64)

jax_func, jax_params = tf2jax.convert(forward, np.zeros_like(p))

jax_forw = jax.jit(jax_func)

num_warmup = 10000
num_samples = 4000
max_sigma = max(min_sigma * 1.1, 0.01)


def bayes_step(y=None):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(params["deg_params_names"]):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    params["deg_" + name + "_min"],
                    params["deg_" + name + "_max"],
                ),
            )
        )
    sigma = numpyro.sample("sigma", dist.Uniform(min_sigma, max_sigma))
    # implement the model
    # needs jax numpy for differentiability here
    y_model, _ = jax_func(jax_params, jnp.array(deg_params))
    std_obs = jnp.ones(y_model.shape[0]) * sigma
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def mcmc_iter(mcmc_method="HMC"):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    # Guess
    theta = []
    for ipar, name in enumerate(params["deg_params_names"]):
        theta.append(
            np.random.uniform(
                params["deg_" + name + "_min"], params["deg_" + name + "_max"]
            )
        )
    theta.append(np.random.uniform(min_sigma, max_sigma))

    # Hamiltonian Monte Carlo (HMC) with no u turn sampling (NUTS)
    if mcmc_method.lower() == "hmc":
        kernel = NUTS(bayes_step, target_accept_prob=0.9)
    elif mcmc_method.lower() == "sa":
        kernel = SA(bayes_step)
    else:
        sys.exit(f"MCMC method {mcmc_method} unrecognized")

    mcmc = MCMC(
        kernel,
        num_chains=1,
        num_warmup=num_warmup,
        num_samples=num_samples,
        jit_model_args=True,
    )
    mcmc.run(rng_key_, y=data_phis_c)
    mcmc.print_summary()

    # Draw samples
    mcmc_samples = mcmc.get_samples()
    labels = list(mcmc_samples.keys())
    nsamples = len(mcmc_samples[labels[0]])
    nparams = len(labels)
    np_mcmc_samples = np.zeros((nsamples, nparams))
    labels_np = params["deg_params_names"] + ["sigma"]
    for ilabel, label in enumerate(labels):
        for ipar, name in enumerate(params["deg_params_names"]):
            if label == name:
                nplabel = labels_np.index(name)
        if label == "sigma":
            nplabel = labels_np.index("sigma")
        np_mcmc_samples[:, nplabel] = np.array(mcmc_samples[label])

    # Uncertainty propagation
    nsamples = np_mcmc_samples.shape[0]
    realization = []
    for i in range(nsamples):
        y = forward(np_mcmc_samples[i, :-1])
        realization.append(y)
    realization = np.array(realization)

    min_real = np.min(realization, axis=0)
    max_real = np.max(realization, axis=0)

    results = {
        "samples": np_mcmc_samples,
        "labels_np": labels_np,
        "labels": labels,
    }

    if (
        np.amin(data_phis_c - min_real) < 0
        or np.amax(data_phis_c - max_real) > 0
    ):
        return False, results
    else:
        return True, results


reduce_sigma, results = mcmc_iter()

np_mcmc_samples = results["samples"]
labels_np = results["labels_np"]
labels = results["labels"]
nparams = len(labels)

figureFolder = "Figures"
os.makedirs(figureFolder, exist_ok=True)


np.savez(
    os.path.join("samp.npz"),
    samples=np_mcmc_samples,
    labels_np=labels_np,
    labels=labels,
)

# Post process
ranges = []
for ipar, name in enumerate(params["deg_params_names"]):
    ranges.append(
        (params["deg_" + name + "_min"], params["deg_" + name + "_max"])
    )
ranges.append((min_sigma, max_sigma))
if deg_param_truth is not None:
    truths = list(deg_param_truth) + [0]
else:
    truths = None
fig = corner.corner(
    np_mcmc_samples, truths=truths, labels=labels_np, bins=50, range=ranges
)
plt.savefig(os.path.join(figureFolder, f"corner.png"))
plt.close()

# Convergence
fig, axes = plt.subplots(nparams, sharex=True)
for i in range(nparams):
    ax = axes[i]
    ax.plot(np_mcmc_samples[:, i], "k", alpha=0.3, rasterized=True)
    ax.set_ylabel(labels[i])
plt.savefig(os.path.join(figureFolder, f"seq.png"))
plt.close()


# Uncertainty propagation
ranget = np.reshape(
    np.linspace(np.amin(data_t), np.amax(data_t), 250), (250, 1)
)
ranget = tf.convert_to_tensor(ranget, dtype=tf.dtypes.float64)
dummyR = params["Rs_c"] * np.ones((ranget.shape[0], 1))
dummyR_tens = (
    tf.convert_to_tensor(dummyR, dtype=tf.dtypes.float64)
    / nn.params["rescale_R"]
)
x = (params["L_a"] + params["L_s"] + params["L_c"]) * np.ones(
    (ranget.shape[0], 1)
)
x_tens = tf.convert_to_tensor(x, dtype=tf.dtypes.float64)
ones_tf64 = tf.ones(tf.shape(ranget), dtype=tf.dtypes.float64)


@tf.function
def forward_range(p, ranget):
    deg_par = [p[i] * ones_tf64 for i in range(len(p))]
    deg_par_resc = [
        nn.rescale_param(p[i], i) * ones_tf64 for i in range(len(p))
    ]
    out = nn.model(
        [
            ranget / params["rescale_T"],
            x_tens / nn.params["rescale_x"],
            dummyR_tens,
        ]
        + deg_par_resc
    )
    phis_c_unrescaled = out[nn.ind_phis_c]
    phis_c = nn.rescalePhis_c(phis_c_unrescaled, ranget, x_tens, *deg_par)[
        :, 0
    ]
    return phis_c


nsamples = np_mcmc_samples.shape[0]
print("Num samples = ", nsamples)
realization = []
for i in range(nsamples):
    y = forward_range(np_mcmc_samples[i, :-1], ranget)
    realization.append(y)
realization = np.array(realization)

mean_real = np.mean(realization, axis=0)
mean_sigma = np.mean(np_mcmc_samples[:, -1])
min_real = np.min(realization, axis=0)
max_real = np.max(realization, axis=0)
std90_real = np.percentile(realization, 90, axis=0)
std10_real = np.percentile(realization, 10, axis=0)

fig = plt.figure()
plt.plot(ranget, mean_real, color="k", linewidth=3, label="mean degradation")
plt.plot(
    ranget,
    std90_real + mean_sigma,
    "--",
    color="k",
    linewidth=3,
    label="10th and 90th percentile",
)
plt.plot(ranget, std10_real - mean_sigma, "--", color="k", linewidth=3)
plt.plot(data_t, data_phis_c, "o", color="r", markersize=7, label="Data")
prettyLabels("time [s]", "phis_c", 14)
plotLegend()
plt.savefig(os.path.join(figureFolder, f"forw.png"))
plt.close()
