import json
import os
import sys
import time

import argument
import numpy as np
import tensorflow as tf
from _losses import (
    loss_fn,
    loss_fn_annealing,
    loss_fn_dynamicAttention_tensor,
    loss_fn_lbfgs,
    loss_fn_lbfgs_annealing,
    loss_fn_lbfgs_SA,
)
from conditionalDecorator import conditional_decorator
from custom_activations import swish_activation
from dataTools import checkDataShape, completeDataset
from eager_lbfgs import Struct, lbfgs
from keras import initializers, layers, losses, optimizers, regularizers
from keras.backend import set_floatx
from keras.callbacks import CSVLogger
from keras.constraints import max_norm, unit_norm
from keras.layers import *
from keras.models import Model
from prettyPlot.progressBar import print_progress_bar

set_floatx("float64")

# Read command line arguments
args = argument.initArg()

if args.optimized:
    optimized = True
else:
    optimized = False


def log(inpt, name):
    if optimized == False:
        print(name)
        print("### min %.2f max %.2f " % (np.amin(inpt), np.amax(inpt)))


def flexible_activation(x, activation):
    if activation == "swish":
        out = Activation(swish_activation)(x)
    elif activation == "sigmoid":
        out = Activation(activation="sigmoid")(x)
    elif activation == "tanh":
        out = Activation(activation="tanh")(x)
    elif activation == "elu":
        out = Activation(activation="elu")(x)
    elif activation == "selu":
        out = Activation(activation="selu")(x)
    elif activation == "gelu":
        out = keras.activations.gelu(x, approximate=True)
    return out


def singleLayer(x, n_units, initializer, activation):
    out = Dense(
        n_units,
        kernel_initializer=initializer,
        bias_initializer=initializer,
    )(x)
    out = flexible_activation(out, activation)
    return out


def pre_resblock(x, pre_unit, res_unit, initializer, activation):
    if not pre_unit == res_unit:
        out = singleLayer(x, res_unit, initializer, activation)
    else:
        out = x
    return out


def resblock(x, n_units, n_layers, initializer, activation):
    tmp_x = x
    for _ in range(n_layers):
        tmp_x = singleLayer(tmp_x, n_units, initializer, activation)
    out = layers.Add()([x, tmp_x])
    out = flexible_activation(out, activation)

    return out


def grad_path(x, n_blocks, n_units, initializer, activation):
    U = singleLayer(x, n_units, initializer, activation)
    V = singleLayer(x, n_units, initializer, activation)

    H = singleLayer(x, n_units, initializer, activation)
    for i_block in range(n_blocks - 1):
        Z = singleLayer(H, n_units, initializer, activation)
        H = (1.0 - Z) * U + Z * V

    return H


def max_tensor_list(tensor_list):
    maxval = np.float64(0.0)
    for i in range(len(tensor_list)):
        if tensor_list[i] is not None:
            maxval = tf.math.maximum(
                (tf.math.reduce_max(tf.math.abs(tensor_list[i]))), maxval
            )
    return maxval


def mean_tensor_list(tensor_list, nTrainablePar):
    meanval = np.float64(0.0)
    ntot = 1e-16
    for i in range(len(tensor_list)):
        if tensor_list[i] is not None:
            meanval = meanval + tf.math.reduce_sum(tf.math.abs(tensor_list[i]))
            ntot = ntot + np.prod(tensor_list[i].shape)
    return meanval / ntot


def safe_save(model, weight_path, overwrite=False):
    saved = False
    ntry = 0
    while not saved:
        try:
            model.save_weights(weight_path, overwrite=overwrite)
            saved = True
        except BlockingIOError:
            ntry += 1
        if ntry > 10000:
            sys.exit(f"ERROR: could not save {weight_path}")


class myNN(Model):
    def __init__(
        self,
        params,
        hidden_units_t=None,
        hidden_units_t_r=None,
        hidden_units_phie=None,
        hidden_units_phis_c=None,
        hidden_units_cs_a=None,
        hidden_units_cs_c=None,
        n_hidden_res_blocks=0,
        n_res_block_layers=1,
        n_res_block_units=1,
        n_grad_path_layers=None,
        n_grad_path_units=None,
        alpha=[0, 0, 0, 0],
        batch_size_int=0,
        batch_size_bound=0,
        max_batch_size_data=0,
        batch_size_reg=0,
        batch_size_struct=64,
        n_batch=0,
        n_batch_lbfgs=0,
        nEpochs_start_lbfgs=10,
        hard_IC_timescale=np.float64(0.81),
        exponentialLimiter=np.float64(10.0),
        collocationMode="fixed",
        gradualTime_sgd=False,
        gradualTime_lbfgs=False,
        firstTime=np.float64(0.1),
        n_gradual_steps_lbfgs=None,
        gradualTimeMode_lbfgs=None,
        tmin_int_bound=np.float64(0.1),
        nEpochs=60,
        nEpochs_lbfgs=60,
        initialLossThreshold=np.float64(100),
        dynamicAttentionWeights=False,
        annealingWeights=False,
        useLossThreshold=True,
        activation="tanh",
        linearizeJ=False,
        lbfgs=False,
        sgd=True,
        params_max=[],
        params_min=[],
        xDataList=[],
        x_params_dataList=[],
        yDataList=[],
        logLossFolder=None,
        modelFolder=None,
        local_utilFolder=None,
        hnn_utilFolder=None,
        hnn_modelFolder=None,
        hnn_params=None,
        hnntime_utilFolder=None,
        hnntime_modelFolder=None,
        hnntime_val=None,
        verbose=False,
        weights=None,
    ):
        Model.__init__(self)

        self.verbose = verbose

        if optimized:
            self.freq = n_batch * 10
        else:
            self.freq = 1

        if logLossFolder is None:
            self.logLossFolder = "Log"
        else:
            self.logLossFolder = logLossFolder
        if modelFolder is None:
            self.modelFolder = "Model"
        else:
            self.modelFolder = modelFolder

        self.params = params
        self.local_utilFolder = local_utilFolder
        self.hnn_utilFolder = hnn_utilFolder
        self.hnn_modelFolder = hnn_modelFolder
        self.hnn_params = hnn_params
        self.use_hnn = False
        if hnn_utilFolder is not None and hnn_modelFolder is not None:
            self.use_hnn = True
            self.vprint("INFO: LOADING HNN...")
            self.vprint(f"\tHNN UTIL FOLDER: {hnn_utilFolder}")
            self.vprint(f"\tHNN MODEL FOLDER: {hnn_modelFolder}")
            if hnn_params is not None:
                self.vprint(f"\tHNN PARAMS: {hnn_params}")

            from load_pinn import load_model

            self.hnn = load_model(
                utilFolder=hnn_utilFolder,
                modelFolder=hnn_modelFolder,
                localUtilFolder=self.local_utilFolder,
                loadDep=True,
                checkRescale=False,
            )
        self.hnntime_utilFolder = hnntime_utilFolder
        self.hnntime_modelFolder = hnntime_modelFolder
        self.hnntime_val = hnntime_val
        self.use_hnntime = False
        if hnntime_utilFolder is not None and hnntime_modelFolder is not None:
            self.use_hnntime = True
            self.vprint("INFO: LOADING HNN-TIME...")
            self.vprint(f"\tHNN-TIME UTIL FOLDER: {hnntime_utilFolder}")
            self.vprint(f"\tHNN-TIME MODEL FOLDER: {hnntime_modelFolder}")
            self.vprint(f"\tHNN-TIME VALUE: {hnntime_val}")
            from load_pinn import load_model

            self.hnntime = load_model(
                utilFolder=hnntime_utilFolder,
                modelFolder=hnntime_modelFolder,
                localUtilFolder=self.local_utilFolder,
                loadDep=True,
                checkRescale=False,
            )
        self.hidden_units_t = hidden_units_t
        self.hidden_units_t_r = hidden_units_t_r
        self.hidden_units_phie = hidden_units_phie
        self.hidden_units_phis_c = hidden_units_phis_c
        self.hidden_units_cs_a = hidden_units_cs_a
        self.hidden_units_cs_c = hidden_units_cs_c
        self.n_hidden_res_blocks = n_hidden_res_blocks
        self.n_res_block_layers = n_res_block_layers
        self.n_res_block_units = n_res_block_units
        self.n_grad_path_layers = n_grad_path_layers
        self.n_grad_path_units = n_grad_path_units
        self.dynamicAttentionWeights = dynamicAttentionWeights
        self.annealingWeights = annealingWeights
        if self.annealingWeights:
            self.dynamicAttentionWeights = False
        self.annealingMaxSet = False
        self.useLossThreshold = useLossThreshold
        if activation.lower() == "swish":
            self.activation = "swish"
        elif activation.lower() == "sigmoid":
            self.activation = "sigmoid"
        elif activation.lower() == "tanh":
            self.activation = "tanh"
        elif activation.lower() == "elu":
            self.activation = "elu"
        elif activation.lower() == "selu":
            self.activation = "selu"
        elif activation.lower() == "gelu":
            self.activation = "gelu"
        else:
            sys.exit("ABORTING: Activation %s unrecognized" % activation)
        self.tmin = np.float64(self.params["tmin"])
        self.tmax = np.float64(self.params["tmax"])
        self.rmin = np.float64(self.params["rmin"])
        self.rmax_a = self.params["Rs_a"]
        self.rmax_c = self.params["Rs_c"]
        self.ind_t = np.int32(0)
        self.ind_r = np.int32(1)
        self.ind_phie = np.int32(0)
        self.ind_phis_c = np.int32(1)
        self.ind_cs_offset = np.int32(2)
        self.ind_cs_a = np.int32(2)
        self.ind_cs_c = np.int32(3)

        self.ind_phie_data = np.int32(0)
        self.ind_phis_c_data = np.int32(1)
        self.ind_cs_offset_data = np.int32(2)
        self.ind_cs_a_data = np.int32(2)
        self.ind_cs_c_data = np.int32(3)

        self.alpha = [np.float64(alphaEntry) for alphaEntry in alpha]
        self.alpha_unweighted = [np.float64(1.0) for alphaEntry in alpha]
        if self.annealingWeights:
            self.alpha = [
                (
                    np.float64(1.0)
                    if np.float64(alphaEntry) > np.float64(1e-12)
                    else np.float64(0.0)
                )
                for alphaEntry in alpha
            ]
        self.phis_a0 = np.float64(0.0)
        self.ce_0 = self.params["ce0"]
        self.cs_a0 = self.params["cs_a0"]
        self.cs_c0 = self.params["cs_c0"]

        # Params
        self.ind_deg_i0_a = np.int32(0)
        self.ind_deg_ds_c = np.int32(1)
        self.ind_deg_i0_a_nn = max(self.ind_t, self.ind_r) + self.ind_deg_i0_a
        self.ind_deg_ds_c_nn = max(self.ind_t, self.ind_r) + self.ind_deg_ds_c
        self.dim_params = np.int32(2)
        self.params_min = params_min
        self.params_max = params_max
        self.resc_params = [
            (min_val + max_val) / 2.0
            for (min_val, max_val) in zip(self.params_min, self.params_max)
        ]

        self.hard_IC_timescale = hard_IC_timescale
        self.exponentialLimiter = exponentialLimiter
        self.collocationMode = collocationMode

        self.firstTime = np.float64(firstTime)
        self.tmin_int_bound = np.float64(tmin_int_bound)
        self.dim_inpt = np.int32(2)

        self.nEpochs = nEpochs

        self.linearizeJ = linearizeJ

        self.gradualTime_sgd = gradualTime_sgd
        self.gradualTime_lbfgs = gradualTime_lbfgs

        # Self dynamics attention weights not allowed with random col
        if self.collocationMode.lower() == "random":
            if self.dynamicAttentionWeights:
                print(
                    "WARNING: dynamic attention weights not allowed with random collocation points"
                )
                print("\tDisabling dynamic attention weights")
            self.dynamicAttentionWeights = False

        if self.gradualTime_sgd:
            self.firstTime = np.float64(firstTime)
            self.timeIncreaseExponent = -np.log(
                (self.firstTime - np.float64(self.params["tmin"]))
                / (
                    np.float64(self.params["tmax"])
                    - np.float64(self.params["tmin"])
                )
            )

        self.reg = 0
        self.n_batch = max(n_batch, 1)
        self.initialLossThreshold = initialLossThreshold

        self.batch_size_int = batch_size_int
        self.batch_size_bound = batch_size_bound
        self.batch_size_reg = batch_size_reg
        self.max_batch_size_data = max_batch_size_data
        if (
            not xDataList == []
            and alpha[2] > 1e-16
            and self.max_batch_size_data > 0
        ):
            # Data points
            for i in range(len(xDataList)):
                checkDataShape(
                    xDataList[i], x_params_dataList[i], yDataList[i]
                )
            # Complete dataset
            ndata = completeDataset(xDataList, x_params_dataList, yDataList)
            # Make tf datasets
            batch_size_data = min(
                ndata // self.n_batch, self.max_batch_size_data
            )
            self.batch_size_data = batch_size_data
            self.new_nData = self.n_batch * self.batch_size_data
            self.vprint("new n data = ", self.new_nData)
            self.vprint("batch_size_data = ", self.batch_size_data)
        self.lbfgs = lbfgs

        # Collocation points
        n_int = n_batch * self.batch_size_int
        n_bound = n_batch * self.batch_size_bound
        n_reg = n_batch * self.batch_size_reg
        tmin_int = self.tmin_int_bound
        tmin_bound = self.tmin_int_bound
        tmin_reg = self.tmin_int_bound

        # figure out which loss to activate
        self.activeInt = True
        self.activeBound = True
        self.activeData = True
        self.activeReg = True
        if self.batch_size_int == 0 or abs(self.alpha[0]) < 1e-12:
            self.vprint("INFO: INT loss is INACTIVE")
            self.activeInt = False
            n_int = n_batch
            self.batch_size_int = 1
        else:
            self.vprint("INFO: INT loss is ACTIVE")
        if self.batch_size_bound == 0 or abs(self.alpha[1]) < 1e-12:
            self.vprint("INFO: BOUND loss is INACTIVE")
            self.activeBound = False
            n_bound = n_batch
            self.batch_size_bound = 1
        else:
            self.vprint("INFO: BOUND loss is ACTIVE")
        if (
            self.max_batch_size_data == 0
            or abs(self.alpha[2]) < 1e-12
            or xDataList == []
        ):
            self.vprint("INFO: DATA loss is INACTIVE")
            self.activeData = False
            n_data = n_batch
            self.batch_size_data = 1
        else:
            self.vprint("INFO: DATA loss is ACTIVE")
        if self.batch_size_reg == 0 or abs(self.alpha[3]) < 1e-12:
            self.vprint("INFO: REG loss is INACTIVE")
            self.activeReg = False
            n_reg = n_batch
            self.batch_size_reg = 1
        else:
            self.vprint("INFO: REG loss is ACTIVE")

        self.setResidualRescaling(weights)

        # Interior loss collocation points
        self.r_a_int = tf.random.uniform(
            (n_int, 1),
            minval=self.rmin + np.float64(1e-12),
            maxval=self.rmax_a,
            dtype=tf.dtypes.float64,
        )
        self.r_c_int = tf.random.uniform(
            (n_int, 1),
            minval=self.rmin + np.float64(1e-12),
            maxval=self.rmax_c,
            dtype=tf.dtypes.float64,
        )
        self.r_maxa_int = self.rmax_a * tf.ones(
            (n_int, 1), dtype=tf.dtypes.float64
        )
        self.r_maxc_int = self.rmax_c * tf.ones(
            (n_int, 1), dtype=tf.dtypes.float64
        )
        if self.gradualTime_sgd:
            self.t_int = tf.random.uniform(
                (n_int, 1),
                minval=tmin_int,
                maxval=self.firstTime,
                dtype=tf.dtypes.float64,
            )
        else:
            self.t_int = tf.random.uniform(
                (n_int, 1),
                minval=tmin_int,
                maxval=self.tmax,
                dtype=tf.dtypes.float64,
            )
        # Params
        self.deg_i0_a_int = tf.random.uniform(
            (n_int, 1),
            minval=self.params["deg_i0_a_min_eff"],
            maxval=self.params["deg_i0_a_max_eff"],
            dtype=tf.dtypes.float64,
        )
        self.deg_ds_c_int = tf.random.uniform(
            (n_int, 1),
            minval=self.params["deg_ds_c_min_eff"],
            maxval=self.params["deg_ds_c_max_eff"],
            dtype=tf.dtypes.float64,
        )

        self.ind_int_col_t = np.int32(0)
        self.ind_int_col_r_a = np.int32(1)
        self.ind_int_col_r_c = np.int32(2)
        self.ind_int_col_r_maxa = np.int32(3)
        self.ind_int_col_r_maxc = np.int32(4)
        self.int_col_pts = [
            self.t_int,
            self.r_a_int,
            self.r_c_int,
            self.r_maxa_int,
            self.r_maxc_int,
        ]
        self.ind_int_col_params_deg_i0_a = np.int32(0)
        self.ind_int_col_params_deg_ds_c = np.int32(1)
        self.int_col_params = [
            self.deg_i0_a_int,
            self.deg_ds_c_int,
        ]

        # Boundary loss collocation points
        self.r_min_bound = tf.zeros((n_bound, 1), dtype=tf.dtypes.float64)
        self.r_maxa_bound = self.rmax_a * tf.ones(
            (n_bound, 1), dtype=tf.dtypes.float64
        )
        self.r_maxc_bound = self.rmax_c * tf.ones(
            (n_bound, 1), dtype=tf.dtypes.float64
        )
        self.deg_i0_a_bound = tf.random.uniform(
            (n_bound, 1),
            minval=self.params["deg_i0_a_min_eff"],
            maxval=self.params["deg_i0_a_max_eff"],
            dtype=tf.dtypes.float64,
        )
        self.deg_ds_c_bound = tf.random.uniform(
            (n_bound, 1),
            minval=self.params["deg_ds_c_min_eff"],
            maxval=self.params["deg_ds_c_max_eff"],
            dtype=tf.dtypes.float64,
        )
        if self.gradualTime_sgd:
            self.t_bound = tf.random.uniform(
                (n_bound, 1),
                minval=tmin_bound,
                maxval=self.firstTime,
                dtype=tf.dtypes.float64,
            )
        else:
            self.t_bound = tf.random.uniform(
                (n_bound, 1),
                minval=tmin_bound,
                maxval=self.tmax,
                dtype=tf.dtypes.float64,
            )

        self.ind_bound_col_t = np.int32(0)
        self.ind_bound_col_r_min = np.int32(1)
        self.ind_bound_col_r_maxa = np.int32(2)
        self.ind_bound_col_r_maxc = np.int32(3)
        self.bound_col_pts = [
            self.t_bound,
            self.r_min_bound,
            self.r_maxa_bound,
            self.r_maxc_bound,
        ]
        self.ind_bound_col_params_deg_i0_a = np.int32(0)
        self.ind_bound_col_params_deg_ds_c = np.int32(1)
        self.bound_col_params = [
            self.deg_i0_a_bound,
            self.deg_ds_c_bound,
        ]

        # Reg loss collocation points
        if self.gradualTime_sgd:
            self.t_reg = tf.random.uniform(
                (n_reg, 1),
                minval=tmin_reg,
                maxval=self.firstTime,
                dtype=tf.dtypes.float64,
            )
        else:
            self.t_reg = tf.random.uniform(
                (n_reg, 1),
                minval=tmin_reg,
                maxval=self.tmax,
                dtype=tf.dtypes.float64,
            )
        self.deg_i0_a_reg = tf.random.uniform(
            (n_reg, 1),
            minval=self.params["deg_i0_a_min_eff"],
            maxval=self.params["deg_i0_a_max_eff"],
            dtype=tf.dtypes.float64,
        )
        self.deg_ds_c_reg = tf.random.uniform(
            (n_reg, 1),
            minval=self.params["deg_ds_c_min_eff"],
            maxval=self.params["deg_ds_c_max_eff"],
            dtype=tf.dtypes.float64,
        )

        self.ind_reg_col_t = np.int32(0)

        self.reg_col_pts = [
            self.t_reg,
        ]
        self.ind_reg_col_params_deg_i0_a = np.int32(0)
        self.ind_reg_col_params_deg_ds_c = np.int32(1)
        self.reg_col_params = [
            self.deg_i0_a_reg,
            self.deg_ds_c_reg,
        ]

        if (
            not self.n_grad_path_layers is None
            and not self.n_grad_path_layers == 0
        ):
            self.makeGradPathModel()
        elif not self.hidden_units_t is None:
            self.makeMergedModel()
        else:
            self.makeSplitModel()

        # Log model
        n_trainable_par = np.sum(
            [np.prod(v._shape) for v in self.model.trainable_variables]
        )
        self.vprint("Num trainable param = ", n_trainable_par)
        self.n_trainable_par = n_trainable_par

        if self.activeData:
            self.xDataList_full = [
                xData[: self.new_nData] for xData in xDataList
            ]
            self.x_params_dataList_full = [
                x_params_data[: self.new_nData]
                for x_params_data in x_params_dataList
            ]
            self.yDataList_full = [
                yData[: self.new_nData] for yData in yDataList
            ]

            ndata_orig = xDataList[0].shape[0]
            if self.new_nData < ndata_orig:
                print(
                    "WARNING: Only %.2f percent of the data will be read"
                    % (100 * self.new_nData / ndata_orig)
                )
                print(
                    "Adjust N_BATCH and MAX_BATCH_SIZE_DATA to accommodate %d datapoints"
                    % ndata_orig
                )
        else:
            self.new_nData = self.n_batch
            self.xDataList_full = [
                (
                    np.zeros((self.n_batch, self.dim_inpt)).astype("float64")
                    if i in [self.ind_cs_a_data, self.ind_cs_c_data]
                    else np.zeros((self.n_batch, self.dim_inpt - 1)).astype(
                        "float64"
                    )
                )
                for i in range(self.n_data_terms)
            ]
            self.x_params_dataList_full = [
                np.zeros((self.n_batch, self.dim_params)).astype("float64")
                for _ in range(self.n_data_terms)
            ]
            self.yDataList_full = [
                np.zeros((self.n_batch, 1)).astype("float64")
                for _ in range(self.n_data_terms)
            ]

        self.n_batch_lbfgs = max(n_batch_lbfgs, 1)
        if self.n_batch_lbfgs > self.n_batch:
            sys.exit("ERROR: n_batch LBFGS must be smaller or equal to SGD's")
        if self.n_batch % self.n_batch_lbfgs > 0:
            sys.exit("ERROR: n_batch SGD must be divisible by LBFGS's")
        self.batch_size_int_lbfgs = int(
            self.batch_size_int * self.n_batch / self.n_batch_lbfgs
        )
        self.batch_size_bound_lbfgs = int(
            self.batch_size_bound * self.n_batch / self.n_batch_lbfgs
        )
        self.batch_size_data_lbfgs = int(
            self.batch_size_data * self.n_batch / self.n_batch_lbfgs
        )
        self.batch_size_reg_lbfgs = int(
            self.batch_size_reg * self.n_batch / self.n_batch_lbfgs
        )
        if self.lbfgs:
            self.nEpochs_lbfgs = nEpochs_lbfgs
            if self.gradualTime_lbfgs:
                self.firstTime = np.float64(firstTime)
                self.n_gradual_steps_lbfgs = int(n_gradual_steps_lbfgs)
                self.nEp_per_gradual_step = self.nEpochs_lbfgs // (
                    2 * self.n_gradual_steps_lbfgs
                )
                self.gradualTimeMode_lbfgs = gradualTimeMode_lbfgs
                self.gradualTimeSchedule_lbfgs = []
                if self.gradualTimeMode_lbfgs.lower() == "linear":
                    for istep_lbfgs in range(self.n_gradual_steps_lbfgs):
                        stepTime = (self.params["tmax"] - self.firstTime) / (
                            self.n_gradual_steps_lbfgs
                        )
                        np.float64(istep_lbfgs) * +self.firstTime
                        new_time_lbfgs = (
                            np.float64(istep_lbfgs) * stepTime + self.firstTime
                        )
                        new_time_lbfgs = min(
                            new_time_lbfgs, self.params["tmax"]
                        )
                        self.gradualTimeSchedule_lbfgs.append(new_time_lbfgs)

                elif self.gradualTimeMode_lbfgs.lower() == "exponential":
                    constantExp_lbfgs = -np.float64(1.0)
                    timeExponent_lbfgs = np.log(
                        self.params["tmax"]
                        - self.firstTime
                        - constantExp_lbfgs
                    ) / np.float64(self.n_gradual_steps_lbfgs)
                    for istep_lbfgs in range(self.n_gradual_steps_lbfgs):
                        new_time_lbfgs = (
                            constantExp_lbfgs
                            + np.exp(timeExponent_lbfgs * istep_lbfgs)
                            + self.firstTime
                        )
                        new_time_lbfgs = min(
                            new_time_lbfgs, self.params["tmax"]
                        )
                        self.gradualTimeSchedule_lbfgs.append(new_time_lbfgs)

            self.nEpochs_start_lbfgs = nEpochs_start_lbfgs
            # Do nEpochs_start_lbfgs iterations at first to make sure the Hessian is correctly computed
            self.nEpochs_lbfgs += self.nEpochs_start_lbfgs
            if self.nEpochs_lbfgs <= self.nEpochs_start_lbfgs:
                self.lbfgs = False
                print(
                    "WARNING: Will not use LBFGS based on number of epoch specified"
                )
            else:
                self.vprint("n_batch_lbfgs = ", self.n_batch_lbfgs)
                self.vprint("n_epoch_lbfgs = ", self.nEpochs_lbfgs)
            self.trueLayers = []
            self.sizes_w = []
            self.sizes_b = []
            for ilayer, layer in enumerate(self.model.layers[0:]):
                weights_biases = layer.get_weights()
                if len(weights_biases) == 0:
                    pass
                else:
                    self.trueLayers.append(ilayer)
                    self.sizes_w.append(len(weights_biases[0].flatten()))
                    self.sizes_b.append(len(weights_biases[1].flatten()))
        self.sgd = sgd
        if self.nEpochs <= 0:
            self.sgd = False
            print(
                "WARNING: Will not use SGD based on number of epoch specified"
            )
            self.dynamicAttentionWeights = False
            self.annealingWeights = False
        else:
            self.vprint("n_batch_sgd = ", self.n_batch)
            self.vprint("n_epoch_sgd = ", self.nEpochs)

        # Make model configuration
        self.config = {}
        self.config["hidden_units_t"] = self.hidden_units_t
        self.config["hidden_units_t_r"] = self.hidden_units_t_r
        self.config["hidden_units_phie"] = self.hidden_units_phie
        self.config["hidden_units_phis_c"] = self.hidden_units_phis_c
        self.config["hidden_units_cs_a"] = self.hidden_units_cs_a
        self.config["hidden_units_cs_c"] = self.hidden_units_cs_c
        self.config["n_hidden_res_blocks"] = self.n_hidden_res_blocks
        self.config["n_res_block_layers"] = self.n_res_block_layers
        self.config["n_res_block_units"] = self.n_res_block_units
        self.config["n_grad_path_layers"] = self.n_grad_path_layers
        self.config["n_grad_path_units"] = self.n_grad_path_units
        self.config["hard_IC_timescale"] = self.hard_IC_timescale
        self.config["exponentialLimiter"] = self.exponentialLimiter
        self.config["dynamicAttentionWeights"] = self.dynamicAttentionWeights
        self.config["annealingWeights"] = self.annealingWeights
        self.config["linearizeJ"] = self.linearizeJ
        self.config["activation"] = self.activation
        self.config["activeInt"] = self.activeInt
        self.config["activeBound"] = self.activeBound
        self.config["activeData"] = self.activeData
        self.config["activeReg"] = self.activeReg
        self.config["params_min"] = self.params_min
        self.config["params_max"] = self.params_max
        self.config["local_utilFolder"] = self.local_utilFolder
        self.config["hnn_utilFolder"] = self.hnn_utilFolder
        self.config["hnn_modelFolder"] = self.hnn_modelFolder
        self.config["hnn_params"] = self.hnn_params

        # Annealing
        self.int_loss_terms = []
        self.bound_loss_terms = []
        self.data_loss_terms = []
        self.reg_loss_terms = []
        self.int_loss_weights = []
        self.bound_loss_weights = []
        self.data_loss_weights = []
        self.reg_loss_weights = []

        if self.annealingWeights:
            self.alpha_anneal = np.float64(0.9 / self.n_batch)
            if self.activeInt:
                n_terms = len(self.interiorTerms_rescale)
                for _ in range(n_terms):
                    self.int_loss_terms.append(
                        tf.Variable(
                            np.float64(0.0), shape=tf.TensorShape(None)
                        )
                    )
                    self.int_loss_weights.append(
                        tf.Variable(
                            np.float64(1.0), shape=tf.TensorShape(None)
                        )
                    )
            else:
                self.int_loss_terms = [
                    tf.Variable(np.float64(0.0), shape=tf.TensorShape(None))
                ]
                self.int_loss_weights = [
                    tf.Variable(np.float64(0.0), shape=tf.TensorShape(None))
                ]

            if self.activeBound:
                n_terms = len(self.boundaryTerms_rescale)
                for _ in range(n_terms):
                    self.bound_loss_terms.append(
                        tf.Variable(
                            np.float64(0.0), shape=tf.TensorShape(None)
                        )
                    )
                    self.bound_loss_weights.append(
                        tf.Variable(
                            np.float64(1.0), shape=tf.TensorShape(None)
                        )
                    )
            else:
                self.bound_loss_terms = [
                    tf.Variable(np.float64(0.0), shape=tf.TensorShape(None))
                ]
                self.bound_loss_weights = [
                    tf.Variable(np.float64(0.0), shape=tf.TensorShape(None))
                ]

            if self.activeData:
                n_terms = len(self.dataTerms_rescale)
                for _ in range(n_terms):
                    self.data_loss_terms.append(
                        tf.Variable(
                            np.float64(0.0), shape=tf.TensorShape(None)
                        )
                    )
                    self.data_loss_weights.append(
                        tf.Variable(
                            np.float64(1.0), shape=tf.TensorShape(None)
                        )
                    )
            else:
                self.data_loss_terms = [
                    tf.Variable(np.float64(0.0), shape=tf.TensorShape(None))
                ]
                self.data_loss_weights = [
                    tf.Variable(np.float64(0.0), shape=tf.TensorShape(None))
                ]

            if self.activeReg:
                n_terms = len(self.regTerms_rescale)
                for _ in range(n_terms):
                    self.reg_loss_terms.append(
                        tf.Variable(
                            np.float64(0.0), shape=tf.TensorShape(None)
                        )
                    )
                    self.reg_loss_weights.append(
                        tf.Variable(
                            np.float64(1.0), shape=tf.TensorShape(None)
                        )
                    )
            else:
                self.reg_loss_terms = [
                    tf.Variable(np.float64(0.0), shape=tf.TensorShape(None))
                ]
                self.reg_loss_weights = [
                    tf.Variable(np.float64(0.0), shape=tf.TensorShape(None))
                ]

        # Dynamic attention
        self.int_col_weights = []
        self.bound_col_weights = []
        self.data_col_weights = []
        self.reg_col_weights = []

        if self.dynamicAttentionWeights:
            self.n_int = self.batch_size_int * self.n_batch
            self.n_bound = self.batch_size_bound * self.n_batch
            self.n_data = self.batch_size_data * self.n_batch
            self.n_reg = self.batch_size_reg * self.n_batch
            if self.activeInt:
                n_terms = len(self.interiorTerms_rescale)
                for _ in range(self.n_batch):
                    tmp = tf.Variable(
                        tf.reshape(
                            tf.repeat(
                                np.float64(1.0), self.batch_size_int * n_terms
                            ),
                            (n_terms, self.batch_size_int, -1),
                        ),
                        trainable=True,
                    )
                    self.int_col_weights.append(tmp)
            else:
                for _ in range(self.n_batch):
                    self.int_col_weights.append([np.float64(0.0)])
            if self.activeBound:
                n_terms = len(self.boundaryTerms_rescale)
                for _ in range(self.n_batch):
                    tmp = tf.Variable(
                        tf.reshape(
                            tf.repeat(
                                np.float64(1.0),
                                self.batch_size_bound * n_terms,
                            ),
                            (n_terms, self.batch_size_bound, -1),
                        ),
                        trainable=True,
                    )
                    self.bound_col_weights.append(tmp)
            else:
                for _ in range(self.n_batch):
                    self.bound_col_weights.append([np.float64(0.0)])
            if self.activeData:
                n_terms = len(self.dataTerms_rescale)
                for _ in range(self.n_batch):
                    tmp = tf.Variable(
                        tf.reshape(
                            tf.repeat(
                                np.float64(1.0), self.batch_size_data * n_terms
                            ),
                            (n_terms, self.batch_size_data, -1),
                        ),
                        trainable=True,
                    )
                    self.data_col_weights.append(tmp)
            else:
                for _ in range(self.n_batch):
                    self.data_col_weights.append([np.float64(0.0)])
            if self.activeReg:
                n_terms = len(self.regTerms_rescale)
                for _ in range(self.n_batch):
                    tmp = tf.Variable(
                        tf.reshape(
                            tf.repeat(
                                np.float64(1.0), self.batch_size_reg * n_terms
                            ),
                            (n_terms, self.batch_size_reg, -1),
                        ),
                        trainable=True,
                    )
                    self.reg_col_weights.append(tmp)
            else:
                for _ in range(self.n_batch):
                    self.reg_col_weights.append([np.float64(0.0)])

    def vprint(self, *kwargs):
        if self.verbose:
            print(*kwargs)

    def makeSplitModel(self):
        self.vprint("INFO: MAKING SPLIT MODEL")

        # inputs
        input_t = Input(shape=(1,), name="input_t")
        input_r = Input(shape=(1,), name="input_r")
        input_deg_i0_a = Input(shape=(1,), name="input_deg_i0_a")
        input_deg_ds_c = Input(shape=(1,), name="input_deg_ds_c")
        input_t_par = concatenate(
            [input_t, input_deg_i0_a, input_deg_ds_c], name="input_t_par"
        )

        if self.use_hnn:
            initializer = "he_normal"
        else:
            initializer = "he_normal"

        # phie
        tmp_phie = input_t_par
        for unit in self.hidden_units_phie:
            tmp_phie = singleLayer(
                tmp_phie, unit, initializer, self.activation
            )
        tmp_phie = pre_resblock(
            tmp_phie,
            self.hidden_units_phie[-1],
            self.n_res_block_units,
            initializer,
            self.activation,
        )
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_phie = resblock(
                tmp_phie,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_phie = Dense(
            1,
            activation="linear",
            name="phie",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )(tmp_phie)

        # phis_c
        tmp_phis_c = input_t_par
        for unit in self.hidden_units_phis_c:
            tmp_phis_c = singleLayer(
                tmp_phis_c, unit, initializer, self.activation
            )

        tmp_phis_c = pre_resblock(
            tmp_phis_c,
            self.hidden_units_phis_c[-1],
            self.n_res_block_units,
            initializer,
            self.activation,
        )
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_phis_c = resblock(
                tmp_phis_c,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_phis_c = Dense(
            1,
            activation="linear",
            name="phis_c",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )(tmp_phis_c)

        # cs_a
        tmp_cs_a = concatenate([input_t_par, input_r], name="input_cs_a")
        for unit in self.hidden_units_cs_a:
            tmp_cs_a = singleLayer(
                tmp_cs_a, unit, initializer, self.activation
            )
        tmp_cs_a = pre_resblock(
            tmp_cs_a,
            self.hidden_units_cs_a[-1],
            self.n_res_block_units,
            initializer,
            self.activation,
        )
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_cs_a = resblock(
                tmp_cs_a,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_cs_a = Dense(
            1,
            activation="linear",
            name="cs_a",
            kernel_initializer=initializer,
            bias_constraint=max_norm(0),
        )(tmp_cs_a)

        # cs_c
        tmp_cs_c = concatenate([input_t_par, input_r], name="input_cs_c")
        for unit in self.hidden_units_cs_c:
            tmp_cs_c = singleLayer(
                tmp_cs_c, unit, initializer, self.activation
            )
        tmp_cs_c = pre_resblock(
            tmp_cs_c,
            self.hidden_units_cs_c[-1],
            self.n_res_block_units,
            initializer,
            self.activation,
        )
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_cs_c = resblock(
                tmp_cs_c,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_cs_c = Dense(
            1,
            activation="linear",
            name="cs_c",
            kernel_initializer=initializer,
            bias_constraint=max_norm(0),
        )(tmp_cs_c)

        self.model = Model(
            [input_t, input_r, input_deg_i0_a, input_deg_ds_c],
            [
                output_phie,
                output_phis_c,
                output_cs_a,
                output_cs_c,
            ],
        )

    def makeMergedModel(self):
        self.vprint("INFO: MAKING MERGED MODEL")

        # inputs
        input_t = Input(shape=(1,), name="input_t")
        input_r = Input(shape=(1,), name="input_r")
        input_deg_i0_a = Input(shape=(1,), name="input_deg_i0_a")
        input_deg_ds_c = Input(shape=(1,), name="input_deg_ds_c")
        input_t_par = concatenate(
            [input_t, input_deg_i0_a, input_deg_ds_c], name="input_t_par"
        )

        if self.use_hnn:
            initializer = "he_normal"
        else:
            initializer = "he_normal"

        # t domain
        tmp_t = input_t_par
        for unit in self.hidden_units_t:
            tmp_t = singleLayer(tmp_t, unit, initializer, self.activation)

        # t_r domain
        tmp_t_r = concatenate([tmp_t, input_r], name="input_t_r")
        for unit in self.hidden_units_t_r:
            tmp_t_r = singleLayer(tmp_t_r, unit, initializer, self.activation)

        # phie
        tmp_phie = tmp_t
        for unit in self.hidden_units_phie:
            tmp_phie = singleLayer(
                tmp_phie, unit, initializer, self.activation
            )

        last_unit = self.hidden_units_t[-1]
        if len(self.hidden_units_phie) > 0:
            last_unit = self.hidden_units_phie[-1]
        tmp_phie = pre_resblock(
            tmp_phie,
            last_unit,
            self.n_res_block_units,
            initializer,
            self.activation,
        )
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_phie = resblock(
                tmp_phie,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_phie = Dense(
            1,
            activation="linear",
            name="phie",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )(tmp_phie)

        # phis_c
        tmp_phis_c = tmp_t
        for unit in self.hidden_units_phis_c:
            tmp_phis_c = singleLayer(
                tmp_phis_c, unit, initializer, self.activation
            )

        last_unit = self.hidden_units_t[-1]
        if len(self.hidden_units_phis_c) > 0:
            last_unit = self.hidden_units_phis_c[-1]
        tmp_phis_c = pre_resblock(
            tmp_phis_c,
            last_unit,
            self.n_res_block_units,
            initializer,
            self.activation,
        )
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_phis_c = resblock(
                tmp_phis_c,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_phis_c = Dense(
            1,
            activation="linear",
            name="phis_c",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )(tmp_phis_c)

        # cs_a
        tmp_cs_a = tmp_t_r
        for unit in self.hidden_units_cs_a:
            tmp_cs_a = singleLayer(
                tmp_cs_a, unit, initializer, self.activation
            )

        last_unit = self.hidden_units_t_r[-1]
        if len(self.hidden_units_cs_a) > 0:
            last_unit = self.hidden_units_cs_a[-1]
        tmp_cs_a = pre_resblock(
            tmp_cs_a,
            last_unit,
            self.n_res_block_units,
            initializer,
            self.activation,
        )
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_cs_a = resblock(
                tmp_cs_a,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_cs_a = Dense(
            1,
            activation="linear",
            name="cs_a",
            kernel_initializer=initializer,
            bias_constraint=max_norm(0),
        )(tmp_cs_a)

        # cs_c
        tmp_cs_c = tmp_t_r
        for unit in self.hidden_units_cs_c:
            tmp_cs_c = singleLayer(
                tmp_cs_c, unit, initializer, self.activation
            )

        last_unit = self.hidden_units_t_r[-1]
        if len(self.hidden_units_cs_c) > 0:
            last_unit = self.hidden_units_cs_c[-1]
        tmp_cs_c = pre_resblock(
            tmp_cs_c,
            last_unit,
            self.n_res_block_units,
            initializer,
            self.activation,
        )
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_cs_c = resblock(
                tmp_cs_c,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_cs_c = Dense(
            1,
            activation="linear",
            name="cs_c",
            kernel_initializer=initializer,
            bias_constraint=max_norm(0),
        )(tmp_cs_c)

        self.model = Model(
            [input_t, input_r, input_deg_i0_a, input_deg_ds_c],
            [
                output_phie,
                output_phis_c,
                output_cs_a,
                output_cs_c,
            ],
        )

    def makeGradPathModel(self):
        self.vprint("INFO: MAKING PATHOLOGY GRADIENT MODEL")

        # inputs
        input_t = Input(shape=(1,), name="input_t")
        input_r = Input(shape=(1,), name="input_r")
        input_deg_i0_a = Input(shape=(1,), name="input_deg_i0_a")
        input_deg_ds_c = Input(shape=(1,), name="input_deg_ds_c")
        input_t_par = concatenate(
            [input_t, input_deg_i0_a, input_deg_ds_c], name="input_t_par"
        )

        if self.use_hnn:
            initializer = "he_normal"
        else:
            initializer = "he_normal"

        # t domain
        tmp_t = input_t_par
        for unit in self.hidden_units_t:
            tmp_t = singleLayer(tmp_t, unit, initializer, self.activation)

        # t_r domain
        tmp_t_r = concatenate([tmp_t, input_r], name="input_t_r")
        for unit in self.hidden_units_t_r:
            tmp_t_r = singleLayer(tmp_t_r, unit, initializer, self.activation)

        # phie
        tmp_phie = tmp_t
        tmp_phie = grad_path(
            tmp_phie,
            self.n_grad_path_layers,
            self.n_grad_path_units,
            initializer,
            self.activation,
        )
        output_phie = Dense(
            1,
            activation="linear",
            name="phie",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )(tmp_phie)

        # phis_c
        tmp_phis_c = tmp_t
        tmp_phis_c = grad_path(
            tmp_phis_c,
            self.n_grad_path_layers,
            self.n_grad_path_units,
            initializer,
            self.activation,
        )
        output_phis_c = Dense(
            1,
            activation="linear",
            name="phis_c",
            kernel_initializer=initializer,
            bias_initializer=initializer,
        )(tmp_phis_c)

        # cs_a
        tmp_cs_a = tmp_t_r
        tmp_cs_a = grad_path(
            tmp_cs_a,
            self.n_grad_path_layers,
            self.n_grad_path_units,
            initializer,
            self.activation,
        )
        output_cs_a = Dense(
            1,
            activation="linear",
            name="cs_a",
            kernel_initializer=initializer,
            bias_initializer=initializer,
            bias_constraint=max_norm(0),
        )(tmp_cs_a)

        # cs_c
        tmp_cs_c = tmp_t_r
        tmp_cs_c = grad_path(
            tmp_cs_c,
            self.n_grad_path_layers,
            self.n_grad_path_units,
            initializer,
            self.activation,
        )
        output_cs_c = Dense(
            1,
            activation="linear",
            name="cs_c",
            kernel_initializer=initializer,
            bias_initializer=initializer,
            bias_constraint=max_norm(0),
        )(tmp_cs_c)

        self.model = Model(
            [input_t, input_r, input_deg_i0_a, input_deg_ds_c],
            [
                output_phie,
                output_phis_c,
                output_cs_a,
                output_cs_c,
            ],
        )

    def loadCol(
        self,
        int_tcol=None,
        int_rcol_a=None,
        int_rcol_c=None,
        int_rcol_maxa=None,
        int_rcol_maxc=None,
        int_weights=None,
        bound_tcol=None,
        bound_rcol_min=None,
        bound_rcol_maxa=None,
        bound_rcol_maxc=None,
        bound_weights=None,
        data_weights=None,
        reg_tcol=None,
        reg_weights=None,
    ):
        raise NotImplementedError
        if self.gradualTime_sgd:
            print("WARNING: Do not load col if you are in gradual time mode")
            print("WARNING: Will not load weights")
            return
        if self.collocationMode == "random":
            print(
                "WARNING: Do not load col if you are in random collocation mode"
            )
            print("WARNING: Will not load weights")
            return
        if self.activeInt:
            self.t_int = tf.convert_to_tensor(int_tcol, dtype=tf.float64)
            self.r_a_int = tf.convert_to_tensor(int_rcol_a, dtype=tf.float64)
            self.r_c_int = tf.convert_to_tensor(int_rcol_c, dtype=tf.float64)
            self.r_maxa_int = tf.convert_to_tensor(
                int_rcol_maxa, dtype=tf.float64
            )
            self.r_maxc_int = tf.convert_to_tensor(
                int_rcol_maxc, dtype=tf.float64
            )
            self.int_col_pts = [
                self.t_int,
                self.r_a_int,
                self.r_c_int,
                self.r_maxa_int,
                self.r_maxc_int,
            ]
        if self.activeReg:
            self.t_reg = tf.convert_to_tensor(reg_tcol, dtype=tf.float64)
            self.reg_col_pts = [
                self.t_reg,
            ]

        if self.activeBound:
            self.t_bound = tf.convert_to_tensor(bound_tcol, dtype=tf.float64)
            self.r_min_bound = tf.convert_to_tensor(
                bound_rcol_min, dtype=tf.float64
            )
            self.r_maxa_bound = tf.convert_to_tensor(
                bound_rcol_maxa, dtype=tf.float64
            )
            self.r_maxc_bound = tf.convert_to_tensor(
                bound_rcol_maxc, dtype=tf.float64
            )
            self.bound_col_pts = [
                self.t_bound,
                self.r_min_bound,
                self.r_maxa_bound,
                self.r_maxc_bound,
            ]
        if self.dynamicAttentionWeights:
            self.n_int = self.batch_size_int * self.n_batch
            self.n_bound = self.batch_size_bound * self.n_batch
            self.n_data = self.batch_size_data * self.n_batch
            self.n_reg = self.batch_size_reg * self.n_batch
            for i in range(self.n_batch):
                if self.activeInt:
                    for j in range(len(self.interiorTerms_rescale)):
                        self.int_col_weights[i][j].assign(
                            np.reshape(
                                int_weights[j][
                                    i
                                    * self.batch_size_int : (i + 1)
                                    * self.batch_size_int
                                ],
                                (self.batch_size_int, 1),
                            )
                        )
                if self.activeBound:
                    for j in range(len(self.boundaryTerms_rescale)):
                        self.bound_col_weights[i][j].assign(
                            np.reshape(
                                bound_weights[j][
                                    i
                                    * self.batch_size_bound : (i + 1)
                                    * self.batch_size_bound
                                ],
                                (self.batch_size_bound, 1),
                            )
                        )
                if self.activeData:
                    for j in range(len(self.dataTerms_rescale)):
                        self.data_col_weights[i][j].assign(
                            np.reshape(
                                data_weights[j][
                                    i
                                    * self.batch_size_data : (i + 1)
                                    * self.batch_size_data
                                ],
                                (self.batch_size_data, 1),
                            )
                        )
                if self.activeReg:
                    for j in range(len(self.regTerms_rescale)):
                        self.reg_col_weights[i][j].assign(
                            np.reshape(
                                reg_weights[j][
                                    i
                                    * self.batch_size_reg : (i + 1)
                                    * self.batch_size_reg
                                ],
                                (self.batch_size_reg, 1),
                            )
                        )

    def set_weights(self, w, sizes_w, sizes_b):
        for i, ind_layer in enumerate(self.trueLayers):
            layer = self.model.layers[ind_layer]
            weights_biases = layer.get_weights()
            shapeW = tf.shape(weights_biases[0])
            shapeB = tf.shape(weights_biases[1])
            start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
            end_weights = sum(sizes_w[: i + 1]) + sum(sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(sizes_w[i] / sizes_b[i])
            weights = tf.reshape(weights, [w_div, sizes_b[i]])
            biases = w[end_weights : end_weights + sizes_b[i]]
            weights_biases = [
                tf.reshape(weights, shapeW),
                tf.reshape(biases, shapeB),
            ]
            layer.set_weights(weights_biases)

    def get_weights(self, model):
        w = []
        for ilayer, layer in enumerate(model.layers[0:]):
            weights_biases = layer.get_weights()
            if len(weights_biases) == 0:
                pass
            else:
                weights = weights_biases[0].flatten()
                biases = weights_biases[1].flatten()
                w.extend(weights)
                w.extend(biases)

        w = tf.convert_to_tensor(w)
        return w

    from _rescale import (
        fix_param,
        get_cs_a_hnn,
        get_cs_a_hnntime,
        get_cs_c_hnn,
        get_cs_c_hnntime,
        get_phie0,
        get_phie_hnn,
        get_phie_hnntime,
        get_phis_c0,
        get_phis_c_hnn,
        get_phis_c_hnntime,
        rescale_param,
        rescaleCs_a,
        rescaleCs_c,
        rescalePhie,
        rescalePhis_c,
        unrescale_param,
    )

    def stretchT(self, t, tmin, tmax, tminp, tmaxp):
        return (t - tmin) * (tmaxp - tminp) / (tmax - tmin) + tminp

    @conditional_decorator(tf.function, optimized)
    def train_step_dynamicAttention(
        self,
        int_col_weights=None,
        bound_col_weights=None,
        data_col_weights=None,
        reg_col_weights=None,
        int_col_pts=None,
        int_col_params=None,
        bound_col_pts=None,
        bound_col_params=None,
        reg_col_pts=None,
        reg_col_params=None,
        x_batch_trainList=None,
        x_cs_batch_trainList=None,
        y_batch_trainList=None,
        x_params_batch_trainList=None,
        x_cs_params_batch_trainList=None,
        tmax=None,
        step=None,
        gradient_threshold=None,
    ):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(int_col_weights)
            tape.watch(bound_col_weights)
            tape.watch(data_col_weights)
            tape.watch(reg_col_weights)

            # get data loss
            interiorTerms = self.interior_loss(
                int_col_pts, int_col_params, tmax
            )
            boundaryTerms = self.boundary_loss(
                bound_col_pts, bound_col_params, tmax
            )
            dataTerms = self.data_loss(
                x_batch_trainList,
                x_cs_batch_trainList,
                x_params_batch_trainList,
                y_batch_trainList,
            )
            regTerms = self.regularization_loss(reg_col_pts, tmax)
            # Rescale residuals
            interiorTerms_rescaled = tf.cast(
                tf.stack(
                    [
                        interiorTerm[0] * resc
                        for (interiorTerm, resc) in zip(
                            interiorTerms, self.interiorTerms_rescale
                        )
                    ],
                    axis=0,
                ),
                dtype=tf.float64,
            )
            boundaryTerms_rescaled = tf.cast(
                tf.stack(
                    [
                        boundaryTerm[0] * resc
                        for (boundaryTerm, resc) in zip(
                            boundaryTerms, self.boundaryTerms_rescale
                        )
                    ],
                    axis=0,
                ),
                dtype=tf.float64,
            )
            dataTerms_rescaled = tf.cast(
                tf.stack(
                    [
                        dataTerm[0] * resc
                        for (dataTerm, resc) in zip(
                            dataTerms, self.dataTerms_rescale
                        )
                    ],
                    axis=0,
                ),
                dtype=tf.float64,
            )

            regTerms_rescaled = tf.cast(
                tf.stack(
                    [
                        regTerm[0] * resc
                        for (regTerm, resc) in zip(
                            regTerms, self.regTerms_rescale
                        )
                    ],
                    axis=0,
                ),
                dtype=tf.float64,
            )
            (
                loss_value,
                loss_value_unweighted,
                int_loss,
                bound_loss,
                data_loss,
                reg_loss,
            ) = loss_fn_dynamicAttention_tensor(
                interiorTerms_rescaled,
                boundaryTerms_rescaled,
                dataTerms_rescaled,
                regTerms_rescaled,
                int_col_weights,
                bound_col_weights,
                data_col_weights,
                reg_col_weights,
                alpha=self.alpha,
            )
            mloss_value = -loss_value

        grads_model = tape.gradient(loss_value, self.model.trainable_weights)
        grads_int_col = tape.gradient(mloss_value, int_col_weights)
        grads_bound_col = tape.gradient(mloss_value, bound_col_weights)
        grads_data_col = tape.gradient(mloss_value, data_col_weights)
        grads_reg_col = tape.gradient(mloss_value, reg_col_weights)

        if gradient_threshold is not None:
            grads_model, glob_norm = tf.clip_by_global_norm(
                grads_model, gradient_threshold
            )
            grads_int_col, _ = tf.clip_by_global_norm(
                grads_int_col, gradient_threshold, use_norm=glob_norm
            )
            grads_bound_col, _ = tf.clip_by_global_norm(
                grads_bound_col, gradient_threshold, use_norm=glob_norm
            )
            grads_data_col, _ = tf.clip_by_global_norm(
                grads_data_col, gradient_threshold, use_norm=glob_norm
            )
            grads_reg_col, _ = tf.clip_by_global_norm(
                grads_reg_col, gradient_threshold, use_norm=glob_norm
            )

        del tape

        return (
            loss_value,
            int_loss / (loss_value),
            bound_loss / (loss_value),
            data_loss / (loss_value),
            reg_loss / (loss_value),
            interiorTerms_rescaled,
            boundaryTerms_rescaled,
            dataTerms_rescaled,
            regTerms_rescaled,
            grads_model,
            grads_int_col,
            grads_bound_col,
            grads_data_col,
            grads_reg_col,
            loss_value_unweighted,
        )

    @conditional_decorator(tf.function, optimized)
    def train_step_annealing(
        self,
        int_col_pts=None,
        int_col_params=None,
        int_loss_weights=None,
        bound_col_pts=None,
        bound_col_params=None,
        bound_loss_weights=None,
        reg_col_pts=None,
        reg_col_params=None,
        reg_loss_weights=None,
        x_batch_trainList=None,
        x_cs_batch_trainList=None,
        x_params_batch_trainList=None,
        y_batch_trainList=None,
        data_loss_weights=None,
        tmax=None,
        gradient_threshold=None,
    ):
        with tf.GradientTape(persistent=True) as tape:
            # get data loss
            interiorTerms = self.interior_loss(
                int_col_pts, int_col_params, tmax
            )
            boundaryTerms = self.boundary_loss(
                bound_col_pts, bound_col_params, tmax
            )
            dataTerms = self.data_loss(
                x_batch_trainList,
                x_cs_batch_trainList,
                x_params_batch_trainList,
                y_batch_trainList,
            )
            regTerms = self.regularization_loss(reg_col_pts, tmax)
            # Rescale residuals
            interiorTerms_rescaled = [
                interiorTerm[0] * resc
                for (interiorTerm, resc) in zip(
                    interiorTerms, self.interiorTerms_rescale
                )
            ]
            boundaryTerms_rescaled = [
                boundaryTerm[0] * resc
                for (boundaryTerm, resc) in zip(
                    boundaryTerms, self.boundaryTerms_rescale
                )
            ]
            dataTerms_rescaled = [
                dataTerm[0] * resc
                for (dataTerm, resc) in zip(dataTerms, self.dataTerms_rescale)
            ]
            regTerms_rescaled = [
                regTerm[0] * resc
                for (regTerm, resc) in zip(regTerms, self.regTerms_rescale)
            ]
            (
                loss_value,
                int_loss,
                bound_loss,
                data_loss,
                reg_loss,
            ) = loss_fn_annealing(
                interiorTerms_rescaled,
                boundaryTerms_rescaled,
                dataTerms_rescaled,
                regTerms_rescaled,
                self.int_loss_terms,
                self.bound_loss_terms,
                self.data_loss_terms,
                self.reg_loss_terms,
                int_loss_weights,
                bound_loss_weights,
                data_loss_weights,
                reg_loss_weights,
                alpha=self.alpha,
            )
        mean_grads_int = [
            mean_tensor_list(
                tape.gradient(
                    self.int_loss_terms[i], self.model.trainable_weights
                ),
                self.n_trainable_par,
            )
            for i in range(len(self.interiorTerms_rescale))
        ]
        max_grads_int = [
            max_tensor_list(
                tape.gradient(
                    self.int_loss_terms[i], self.model.trainable_weights
                )
            )
            for i in range(len(self.interiorTerms_rescale))
        ]
        mean_grads_bound = [
            mean_tensor_list(
                tape.gradient(
                    self.bound_loss_terms[i], self.model.trainable_weights
                ),
                self.n_trainable_par,
            )
            for i in range(len(self.boundaryTerms_rescale))
        ]
        max_grads_bound = [
            max_tensor_list(
                tape.gradient(
                    self.bound_loss_terms[i], self.model.trainable_weights
                )
            )
            for i in range(len(self.boundaryTerms_rescale))
        ]
        mean_grads_data = [
            mean_tensor_list(
                tape.gradient(
                    self.data_loss_terms[i], self.model.trainable_weights
                ),
                self.n_trainable_par,
            )
            for i in range(len(self.dataTerms_rescale))
        ]
        max_grads_data = [
            max_tensor_list(
                tape.gradient(
                    self.data_loss_terms[i], self.model.trainable_weights
                )
            )
            for i in range(len(self.dataTerms_rescale))
        ]
        mean_grads_reg = [
            mean_tensor_list(
                tape.gradient(
                    self.reg_loss_terms[i], self.model.trainable_weights
                ),
                self.n_trainable_par,
            )
            for i in range(len(self.regTerms_rescale))
        ]
        max_grads_reg = [
            max_tensor_list(
                tape.gradient(
                    self.reg_loss_terms[i], self.model.trainable_weights
                )
            )
            for i in range(len(self.regTerms_rescale))
        ]

        grads_model = tape.gradient(loss_value, self.model.trainable_weights)

        if gradient_threshold is not None:
            grads_model, glob_norm = tf.clip_by_global_norm(
                grads_model, gradient_threshold
            )
            max_grads_int, _ = tf.clip_by_global_norm(
                max_grads_int, gradient_threshold, use_norm=glob_norm
            )
            max_grads_bound, _ = tf.clip_by_global_norm(
                max_grads_bound, gradient_threshold, use_norm=glob_norm
            )
            max_grads_data, _ = tf.clip_by_global_norm(
                max_grads_data, gradient_threshold, use_norm=glob_norm
            )
            max_grads_reg, _ = tf.clip_by_global_norm(
                max_grads_reg, gradient_threshold, use_norm=glob_norm
            )
            mean_grads_int, _ = tf.clip_by_global_norm(
                mean_grads_int, gradient_threshold, use_norm=glob_norm
            )
            mean_grads_bound, _ = tf.clip_by_global_norm(
                mean_grads_bound, gradient_threshold, use_norm=glob_norm
            )
            mean_grads_data, _ = tf.clip_by_global_norm(
                mean_grads_data, gradient_threshold, use_norm=glob_norm
            )
            mean_grads_reg, _ = tf.clip_by_global_norm(
                mean_grads_reg, gradient_threshold, use_norm=glob_norm
            )

        return (
            loss_value,
            int_loss / (loss_value),
            bound_loss / (loss_value),
            data_loss / (loss_value),
            reg_loss / (loss_value),
            interiorTerms_rescaled,
            boundaryTerms_rescaled,
            dataTerms_rescaled,
            regTerms_rescaled,
            grads_model,
            max_grads_int,
            max_grads_bound,
            max_grads_data,
            max_grads_reg,
            mean_grads_int,
            mean_grads_bound,
            mean_grads_data,
            mean_grads_reg,
        )

    @conditional_decorator(tf.function, optimized)
    def train_step(
        self,
        int_col_pts=None,
        int_col_params=None,
        bound_col_pts=None,
        bound_col_params=None,
        reg_col_pts=None,
        reg_col_params=None,
        x_batch_trainList=None,
        x_cs_batch_trainList=None,
        x_params_batch_trainList=None,
        y_batch_trainList=None,
        tmax=None,
        gradient_threshold=None,
    ):
        with tf.GradientTape(persistent=True) as tape:
            # get data loss
            interiorTerms = self.interior_loss(
                int_col_pts, int_col_params, tmax
            )
            boundaryTerms = self.boundary_loss(
                bound_col_pts, bound_col_params, tmax
            )
            dataTerms = self.data_loss(
                x_batch_trainList,
                x_cs_batch_trainList,
                x_params_batch_trainList,
                y_batch_trainList,
            )
            regTerms = self.regularization_loss(reg_col_pts, tmax)
            # Rescale residuals
            interiorTerms_rescaled = [
                interiorTerm[0] * resc
                for (interiorTerm, resc) in zip(
                    interiorTerms, self.interiorTerms_rescale
                )
            ]
            boundaryTerms_rescaled = [
                boundaryTerm[0] * resc
                for (boundaryTerm, resc) in zip(
                    boundaryTerms, self.boundaryTerms_rescale
                )
            ]
            dataTerms_rescaled = [
                dataTerm[0] * resc
                for (dataTerm, resc) in zip(dataTerms, self.dataTerms_rescale)
            ]
            regTerms_rescaled = [
                regTerm[0] * resc
                for (regTerm, resc) in zip(regTerms, self.regTerms_rescale)
            ]
            loss_value, int_loss, bound_loss, data_loss, reg_loss = loss_fn(
                interiorTerms_rescaled,
                boundaryTerms_rescaled,
                dataTerms_rescaled,
                regTerms_rescaled,
                alpha=self.alpha,
            )
        grads_model = tape.gradient(loss_value, self.model.trainable_weights)
        if gradient_threshold is not None:
            grads_model, glob_norm = tf.clip_by_global_norm(
                grads_model, gradient_threshold
            )

        return (
            loss_value,
            int_loss / (loss_value),
            bound_loss / (loss_value),
            data_loss / (loss_value),
            reg_loss / (loss_value),
            interiorTerms_rescaled,
            boundaryTerms_rescaled,
            dataTerms_rescaled,
            regTerms_rescaled,
            grads_model,
        )

    @conditional_decorator(tf.function, optimized)
    def applyGrad_dynamicAttention_int(
        self,
        grads,
        weights,
    ):
        self.dsOptimizerInt.apply_gradients(zip(grads, weights))

    @conditional_decorator(tf.function, optimized)
    def applyGrad_dynamicAttention_bound(
        self,
        grads,
        weights,
    ):
        self.dsOptimizerBound.apply_gradients(zip(grads, weights))

    @conditional_decorator(tf.function, optimized)
    def applyGrad_dynamicAttention_data(
        self,
        grads,
        weights,
    ):
        self.dsOptimizerData.apply_gradients(zip(grads, weights))

    @conditional_decorator(tf.function, optimized)
    def applyGrad_dynamicAttention_reg(
        self,
        grads,
        weights,
    ):
        self.dsOptimizerReg.apply_gradients(zip(grads, weights))

    @conditional_decorator(tf.function, optimized)
    def applyGrad(self, grads_model):
        self.dsOptimizerModel.apply_gradients(
            zip(grads_model, self.model.trainable_weights)
        )

    from _losses import (
        boundary_loss,
        data_loss,
        get_loss_and_flat_grad,
        get_loss_and_flat_grad_annealing,
        get_loss_and_flat_grad_SA,
        get_unweighted_loss,
        interior_loss,
        regularization_loss,
        setResidualRescaling,
    )
    from dataTools import check_loss_dim

    def runLBFGS(
        self,
        tmax,
        nIter,
        epochDoneLBFGS,
        epochDoneSGD,
        bestLoss,
        learningRateLBFGS,
        gradient_threshold=None,
    ):
        if self.dynamicAttentionWeights and (
            tmax is None or abs(self.params["tmax"] - tmax) < 1e-12
        ):
            loss_and_flat_grad = self.get_loss_and_flat_grad_SA(
                self.int_col_pts,
                self.int_col_params,
                self.bound_col_pts,
                self.bound_col_params,
                self.reg_col_pts,
                self.reg_col_params,
                self.flatIntColWeights,
                self.flatBoundColWeights,
                self.flatDataColWeights,
                self.flatRegColWeights,
                [tf.convert_to_tensor(x) for x in self.xDataList_full],
                [tf.convert_to_tensor(x) for x in self.x_params_dataList_full],
                [tf.convert_to_tensor(y) for y in self.yDataList_full],
                self.n_batch_lbfgs,
                tmax=tmax,
                gradient_threshold=gradient_threshold,
            )
        elif (
            self.dynamicAttentionWeights
            and abs(self.params["tmax"] - tmax) >= 1e-12
        ):
            loss_and_flat_grad = self.get_loss_and_flat_grad_SA(
                self.int_col_pts,
                self.int_col_params,
                self.bound_col_pts,
                self.bound_col_params,
                self.reg_col_pts,
                self.reg_col_params,
                self.flatIntColOnes,
                self.flatBoundColOnes,
                self.flatDataColOnes,
                self.flatRegColOnes,
                [tf.convert_to_tensor(x) for x in self.xDataList_full],
                [tf.convert_to_tensor(x) for x in self.x_params_dataList_full],
                [tf.convert_to_tensor(y) for y in self.yDataList_full],
                self.n_batch_lbfgs,
                tmax=tmax,
                gradient_threshold=gradient_threshold,
            )
        elif self.annealingWeights:
            loss_and_flat_grad = self.get_loss_and_flat_grad_annealing(
                self.int_col_pts,
                self.int_col_params,
                self.int_loss_weights,
                self.bound_col_pts,
                self.bound_col_params,
                self.bound_loss_weights,
                self.reg_col_pts,
                self.reg_col_params,
                self.reg_loss_weights,
                [tf.convert_to_tensor(x) for x in self.xDataList_full],
                [tf.convert_to_tensor(x) for x in self.x_params_dataList_full],
                [tf.convert_to_tensor(y) for y in self.yDataList_full],
                self.data_loss_weights,
                self.n_batch_lbfgs,
                tmax=tmax,
                gradient_threshold=gradient_threshold,
            )
        else:
            loss_and_flat_grad = self.get_loss_and_flat_grad(
                self.int_col_pts,
                self.int_col_params,
                self.bound_col_pts,
                self.bound_col_params,
                self.reg_col_pts,
                self.reg_col_params,
                [tf.convert_to_tensor(x) for x in self.xDataList_full],
                [tf.convert_to_tensor(x) for x in self.x_params_dataList_full],
                [tf.convert_to_tensor(y) for y in self.yDataList_full],
                self.n_batch_lbfgs,
                tmax=tmax,
                gradient_threshold=gradient_threshold,
            )

        _, _, _, bestLoss = lbfgs(
            loss_and_flat_grad,
            self.get_weights(self.model),
            Struct(),
            model=self.model,
            bestLoss=bestLoss,
            modelFolder=self.modelFolder,
            maxIter=nIter,
            learningRate=learningRateLBFGS,
            dynamicAttention=self.dynamicAttentionWeights,
            logLossFolder=self.logLossFolder,
            nEpochDoneLBFGS=epochDoneLBFGS,
            nEpochDoneSGD=epochDoneSGD,
            nBatchSGD=self.n_batch,
            nEpochs_start_lbfgs=self.nEpochs_start_lbfgs,
        )

        return bestLoss

    def train(
        self,
        learningRateModel,
        learningRateModelFinal,
        lrSchedulerModel,
        learningRateWeights=None,
        learningRateWeightsFinal=None,
        lrSchedulerWeights=None,
        learningRateLBFGS=None,
        inner_epochs=None,
        start_weight_training_epoch=None,
        gradient_threshold=None,
    ):
        if gradient_threshold is not None:
            print(f"INFO: clipping gradients at {gradient_threshold:.2g}")
        # Make sure the control file for learning rate is consistent with main.py, at least at first
        self.prepareLog()
        self.initLearningRateControl(learningRateModel, learningRateWeights)
        bestLoss = None
        # Make sure the control file for loss threshold is consistent with main.py, at least at first
        self.initLossThresholdControl(self.initialLossThreshold)

        train_dataset_phie = tf.data.Dataset.from_tensor_slices(
            (
                np.reshape(
                    self.xDataList_full[self.ind_phie_data],
                    (self.new_nData, self.dim_inpt - 1),
                ),
                np.reshape(
                    self.x_params_dataList_full[self.ind_phie_data],
                    (self.new_nData, self.dim_params),
                ),
                np.reshape(
                    self.yDataList_full[self.ind_phie_data],
                    (self.new_nData, 1),
                ),
            )
        ).batch(self.batch_size_data)
        train_dataset_phis_c = tf.data.Dataset.from_tensor_slices(
            (
                np.reshape(
                    self.xDataList_full[self.ind_phis_c_data],
                    (self.new_nData, self.dim_inpt - 1),
                ),
                np.reshape(
                    self.x_params_dataList_full[self.ind_phis_c_data],
                    (self.new_nData, self.dim_params),
                ),
                np.reshape(
                    self.yDataList_full[self.ind_phis_c_data],
                    (self.new_nData, 1),
                ),
            )
        ).batch(self.batch_size_data)
        train_dataset_cs_a = tf.data.Dataset.from_tensor_slices(
            (
                np.reshape(
                    self.xDataList_full[self.ind_cs_a_data],
                    (self.new_nData, self.dim_inpt),
                ),
                np.reshape(
                    self.x_params_dataList_full[self.ind_cs_a_data],
                    (self.new_nData, self.dim_params),
                ),
                np.reshape(
                    self.yDataList_full[self.ind_cs_a_data],
                    (self.new_nData, 1),
                ),
            )
        ).batch(self.batch_size_data)
        train_dataset_cs_c = tf.data.Dataset.from_tensor_slices(
            (
                np.reshape(
                    self.xDataList_full[self.ind_cs_c_data],
                    (self.new_nData, self.dim_inpt),
                ),
                np.reshape(
                    self.x_params_dataList_full[self.ind_cs_c_data],
                    (self.new_nData, self.dim_params),
                ),
                np.reshape(
                    self.yDataList_full[self.ind_cs_c_data],
                    (self.new_nData, 1),
                ),
            )
        ).batch(self.batch_size_data)

        # INTERIOR
        train_dataset_t_int = tf.data.Dataset.from_tensor_slices(
            self.t_int
        ).batch(self.batch_size_int)
        train_dataset_r_a_int = tf.data.Dataset.from_tensor_slices(
            self.r_a_int
        ).batch(self.batch_size_int)
        train_dataset_r_c_int = tf.data.Dataset.from_tensor_slices(
            self.r_c_int
        ).batch(self.batch_size_int)
        train_dataset_r_maxa_int = tf.data.Dataset.from_tensor_slices(
            self.r_maxa_int
        ).batch(self.batch_size_int)
        train_dataset_r_maxc_int = tf.data.Dataset.from_tensor_slices(
            self.r_maxc_int
        ).batch(self.batch_size_int)

        train_dataset_deg_i0_a_int = tf.data.Dataset.from_tensor_slices(
            self.deg_i0_a_int
        ).batch(self.batch_size_int)
        train_dataset_deg_ds_c_int = tf.data.Dataset.from_tensor_slices(
            self.deg_ds_c_int
        ).batch(self.batch_size_int)

        # BOUNDARY
        train_dataset_t_bound = tf.data.Dataset.from_tensor_slices(
            self.t_bound
        ).batch(self.batch_size_bound)
        train_dataset_r_min_bound = tf.data.Dataset.from_tensor_slices(
            self.r_min_bound
        ).batch(self.batch_size_bound)
        train_dataset_r_maxa_bound = tf.data.Dataset.from_tensor_slices(
            self.r_maxa_bound
        ).batch(self.batch_size_bound)
        train_dataset_r_maxc_bound = tf.data.Dataset.from_tensor_slices(
            self.r_maxc_bound
        ).batch(self.batch_size_bound)

        train_dataset_deg_i0_a_bound = tf.data.Dataset.from_tensor_slices(
            self.deg_i0_a_bound
        ).batch(self.batch_size_bound)
        train_dataset_deg_ds_c_bound = tf.data.Dataset.from_tensor_slices(
            self.deg_ds_c_bound
        ).batch(self.batch_size_bound)

        # REG
        train_dataset_t_reg = tf.data.Dataset.from_tensor_slices(
            self.t_reg
        ).batch(self.batch_size_reg)
        train_dataset_deg_i0_a_reg = tf.data.Dataset.from_tensor_slices(
            self.deg_i0_a_reg
        ).batch(self.batch_size_reg)
        train_dataset_deg_ds_c_reg = tf.data.Dataset.from_tensor_slices(
            self.deg_ds_c_reg
        ).batch(self.batch_size_reg)

        # Prepare LR
        lr_m = learningRateModel
        self.dsOptimizerModel = optimizers.Adam(learning_rate=lr_m)
        if self.dynamicAttentionWeights:
            lr_w = learningRateWeights
            self.dsOptimizerInt = optimizers.Adam(learning_rate=lr_w)
            self.dsOptimizerBound = optimizers.Adam(learning_rate=lr_w)
            self.dsOptimizerData = optimizers.Adam(learning_rate=lr_w)
            self.dsOptimizerReg = optimizers.Adam(learning_rate=lr_w)
        # Prepare Loss threshold
        lt = self.initialLossThreshold

        # Collocation Log
        print("Using collocation points: " + self.collocationMode)

        # Train
        print_progress_bar(
            0,
            self.nEpochs,
            prefix="Loss=%s  Epoch= %d / %d " % ("?", 0, self.nEpochs),
            suffix="Complete",
            length=20,
        )
        true_epoch_num = 0
        self.printed_start_SA = False

        self.run_SGD = True
        self.run_LBFGS = False

        for epoch in range(self.nEpochs):
            if epoch > 0:
                old_mse_loss = train_mse_loss
                old_mse_unweighted_loss = train_mse_unweighted_loss
            train_mse_loss = 0
            train_mse_unweighted_loss = 0
            time_per_step = 0
            self.lr_m_epoch_start = lr_m
            if self.dynamicAttentionWeights:
                self.lr_w_epoch_start = lr_w
            if self.useLossThreshold:
                self.lt_epoch_start = lt
            if self.gradualTime_sgd:
                if self.nEpochs > 3:
                    new_tmax = (
                        (
                            np.float64(self.params["tmax"])
                            - np.float64(self.params["tmin"])
                        )
                        * np.float64(
                            np.exp(
                                self.timeIncreaseExponent
                                * ((epoch) / (self.nEpochs // 2 - 1) - 1)
                            )
                        )
                    ) + np.float64(self.params["tmin"])
                    new_tmax = min(new_tmax, self.params["tmax"])
                else:
                    new_tmax = self.params["tmax"]
                if abs(new_tmax - self.params["tmax"]) > 1e-12:
                    print("")
                    print("tmax = ", new_tmax)
                    print("")
                    if epoch >= start_weight_training_epoch:
                        start_weight_training_epoch += 1
            else:
                new_tmax = None

            currentEpoch_it = 0
            loss_increase_occurence = 0
            while True:
                currentEpoch_it += 1
                tot_grad_int = []
                tot_grad_bound = []
                tot_grad_data = []
                tot_grad_reg = []

                for step, (
                    t_int_col_pts,
                    r_a_int_col_pts,
                    r_c_int_col_pts,
                    r_maxa_int_col_pts,
                    r_maxc_int_col_pts,
                    deg_i0_a_int_col_params,
                    deg_ds_c_int_col_params,
                    t_bound_col_pts,
                    r_min_bound_col_pts,
                    r_maxa_bound_col_pts,
                    r_maxc_bound_col_pts,
                    deg_i0_a_bound_col_params,
                    deg_ds_c_bound_col_params,
                    t_reg_col_pts,
                    deg_i0_a_reg_col_params,
                    deg_ds_c_reg_col_params,
                    (
                        x_batch_train_phie,
                        x_params_batch_train_phie,
                        y_batch_train_phie,
                    ),
                    (
                        x_batch_train_phis_c,
                        x_params_batch_train_phis_c,
                        y_batch_train_phis_c,
                    ),
                    (
                        x_batch_train_cs_a,
                        x_params_batch_train_cs_a,
                        y_batch_train_cs_a,
                    ),
                    (
                        x_batch_train_cs_c,
                        x_params_batch_train_cs_c,
                        y_batch_train_cs_c,
                    ),
                ) in enumerate(
                    zip(
                        train_dataset_t_int,
                        train_dataset_r_a_int,
                        train_dataset_r_c_int,
                        train_dataset_r_maxa_int,
                        train_dataset_r_maxc_int,
                        train_dataset_deg_i0_a_int,
                        train_dataset_deg_ds_c_int,
                        train_dataset_t_bound,
                        train_dataset_r_min_bound,
                        train_dataset_r_maxa_bound,
                        train_dataset_r_maxc_bound,
                        train_dataset_deg_i0_a_bound,
                        train_dataset_deg_ds_c_bound,
                        train_dataset_t_reg,
                        train_dataset_deg_i0_a_reg,
                        train_dataset_deg_ds_c_reg,
                        train_dataset_phie,
                        train_dataset_phis_c,
                        train_dataset_cs_a,
                        train_dataset_cs_c,
                    )
                ):
                    self.total_step = step + true_epoch_num * self.n_batch
                    self.step = step

                    x_batch_trainList = tf.stack(
                        [
                            x_batch_train_phie,
                            x_batch_train_phis_c,
                        ],
                        axis=0,
                    )

                    x_cs_batch_trainList = tf.stack(
                        [
                            x_batch_train_cs_a,
                            x_batch_train_cs_c,
                        ],
                        axis=0,
                    )

                    x_params_batch_trainList = tf.stack(
                        [
                            x_params_batch_train_phie,
                            x_params_batch_train_phis_c,
                            x_params_batch_train_cs_a,
                            x_params_batch_train_cs_c,
                        ],
                        axis=0,
                    )
                    y_batch_trainList = tf.stack(
                        [
                            y_batch_train_phie,
                            y_batch_train_phis_c,
                            y_batch_train_cs_a,
                            y_batch_train_cs_c,
                        ],
                        axis=0,
                    )

                    int_col_pts = tf.stack(
                        [
                            t_int_col_pts,
                            r_a_int_col_pts,
                            r_c_int_col_pts,
                            r_maxa_int_col_pts,
                            r_maxc_int_col_pts,
                        ],
                        axis=0,
                    )
                    int_col_params = tf.stack(
                        [
                            deg_i0_a_int_col_params,
                            deg_ds_c_int_col_params,
                        ],
                        axis=0,
                    )

                    bound_col_pts = tf.stack(
                        [
                            t_bound_col_pts,
                            r_min_bound_col_pts,
                            r_maxa_bound_col_pts,
                            r_maxc_bound_col_pts,
                        ],
                        axis=0,
                    )

                    bound_col_params = tf.stack(
                        [
                            deg_i0_a_bound_col_params,
                            deg_ds_c_bound_col_params,
                        ],
                        axis=0,
                    )

                    reg_col_pts = tf.stack(
                        [
                            t_reg_col_pts,
                        ],
                        axis=0,
                    )

                    reg_col_params = tf.stack(
                        [
                            deg_i0_a_reg_col_params,
                            deg_ds_c_reg_col_params,
                        ],
                        axis=0,
                    )

                    if self.dynamicAttentionWeights:
                        time_s = time.time()
                        tf_step = tf.constant(step)
                        sliced_int_col_weights = tf.gather(
                            self.int_col_weights, tf_step
                        )
                        sliced_bound_col_weights = tf.gather(
                            self.bound_col_weights, tf_step
                        )
                        sliced_data_col_weights = tf.gather(
                            self.data_col_weights, tf_step
                        )
                        sliced_reg_col_weights = tf.gather(
                            self.reg_col_weights, tf_step
                        )
                        loss_info = self.train_step_dynamicAttention(
                            x_batch_trainList=x_batch_trainList,
                            x_cs_batch_trainList=x_cs_batch_trainList,
                            x_params_batch_trainList=x_params_batch_trainList,
                            y_batch_trainList=y_batch_trainList,
                            int_col_pts=int_col_pts,
                            int_col_params=int_col_params,
                            bound_col_pts=bound_col_pts,
                            bound_col_params=bound_col_params,
                            reg_col_pts=reg_col_pts,
                            reg_col_params=reg_col_params,
                            int_col_weights=sliced_int_col_weights,
                            bound_col_weights=sliced_bound_col_weights,
                            data_col_weights=sliced_data_col_weights,
                            reg_col_weights=sliced_reg_col_weights,
                            tmax=new_tmax,
                            gradient_threshold=gradient_threshold,
                        )
                        (
                            grads_model,
                            grads_int_col,
                            grads_bound_col,
                            grads_data_col,
                            grads_reg_col,
                        ) = loss_info[9:14]

                        tot_grad_int.append(grads_int_col)
                        tot_grad_bound.append(grads_bound_col)
                        tot_grad_data.append(grads_data_col)
                        tot_grad_reg.append(grads_reg_col)

                    elif self.annealingWeights:
                        time_s = time.time()
                        loss_info = self.train_step_annealing(
                            x_batch_trainList=x_batch_trainList,
                            x_cs_batch_trainList=x_cs_batch_trainList,
                            x_params_batch_trainList=x_params_batch_trainList,
                            y_batch_trainList=y_batch_trainList,
                            data_loss_weights=self.data_loss_weights,
                            int_col_pts=int_col_pts,
                            int_col_params=int_col_params,
                            int_loss_weights=self.int_loss_weights,
                            bound_col_pts=bound_col_pts,
                            bound_col_params=bound_col_params,
                            bound_loss_weights=self.bound_loss_weights,
                            reg_col_pts=reg_col_pts,
                            reg_col_params=reg_col_params,
                            reg_loss_weights=self.reg_loss_weights,
                            tmax=new_tmax,
                            gradient_threshold=gradient_threshold,
                        )
                        grads_model = loss_info[9]
                        if epoch >= start_weight_training_epoch:
                            max_grad_int = loss_info[10]
                            max_grad_bound = loss_info[11]
                            max_grad_data = loss_info[12]
                            max_grad_reg = loss_info[13]
                            mean_grads_int = loss_info[14]
                            mean_grads_bound = loss_info[15]
                            mean_grads_data = loss_info[16]
                            mean_grads_reg = loss_info[17]
                            allmax = []
                            if self.activeInt:
                                allmax += max_grad_int
                            if self.activeBound:
                                allmax += max_grad_bound
                            if self.activeData:
                                allmax += max_grad_data
                            if self.activeReg:
                                allmax += max_grad_reg
                            if not self.annealingMaxSet:
                                # Find smallest maximum
                                max_grad_id = np.argmax(allmax)
                                max_grad_int_id = np.argmax(max_grad_int)
                                max_grad_bound_id = np.argmax(max_grad_bound)
                                max_grad_data_id = np.argmax(max_grad_data)
                                max_grad_reg_id = np.argmax(max_grad_reg)
                                int_ref = False
                                bound_ref = False
                                data_ref = False
                                reg_ref = False
                                if self.activeInt and abs(
                                    allmax[max_grad_id]
                                    - max_grad_int[max_grad_int_id]
                                ) < np.float64(1e-12):
                                    int_ref = True
                                if self.activeBound and abs(
                                    allmax[max_grad_id]
                                    - max_grad_bound[max_grad_bound_id]
                                ) < np.float64(1e-12):
                                    bound_ref = True
                                if self.activeData and abs(
                                    allmax[max_grad_id]
                                    - max_grad_data[max_grad_data_id]
                                ) < np.float64(1e-12):
                                    data_ref = True
                                if self.activeReg and abs(
                                    allmax[max_grad_id]
                                    - max_grad_reg[max_grad_reg_id]
                                ) < np.float64(1e-12):
                                    reg_ref = True
                                self.annealingMaxSet = True
                            maxGradRef = allmax[max_grad_id]
                            for i in range(len(self.interiorTerms_rescale)):
                                if int_ref and i == max_grad_int_id:
                                    continue
                                self.int_loss_weights[i] = np.clip(
                                    self.int_loss_weights[i]
                                    * (np.float64(1.0) - self.alpha_anneal)
                                    + self.alpha_anneal
                                    * maxGradRef
                                    / (mean_grads_int[i] + np.float64(1e-16)),
                                    a_min=np.float64(1e-1),
                                    a_max=np.float64(1e6),
                                )
                            for i in range(len(self.boundaryTerms_rescale)):
                                if bound_ref and i == max_grad_bound_id:
                                    continue
                                self.bound_loss_weights[i] = np.clip(
                                    self.bound_loss_weights[i]
                                    * (np.float64(1.0) - self.alpha_anneal)
                                    + self.alpha_anneal
                                    * maxGradRef
                                    / (
                                        mean_grads_bound[i] + np.float64(1e-16)
                                    ),
                                    a_min=np.float64(1e-1),
                                    a_max=np.float64(1e6),
                                )
                            for i in range(len(self.dataTerms_rescale)):
                                if data_ref and i == max_grad_data_id:
                                    continue
                                self.data_loss_weights[i] = np.clip(
                                    self.data_loss_weights[i]
                                    * (np.float64(1.0) - self.alpha_anneal)
                                    + self.alpha_anneal
                                    * maxGradRef
                                    / (mean_grads_data[i] + np.float64(1e-16)),
                                    a_min=np.float64(1e-1),
                                    a_max=np.float64(1e6),
                                )
                            for i in range(len(self.regTerms_rescale)):
                                if reg_ref and i == max_grad_reg_id:
                                    continue
                                self.reg_loss_weights[i] = np.clip(
                                    self.reg_loss_weights[i]
                                    * (np.float64(1.0) - self.alpha_anneal)
                                    + self.alpha_anneal
                                    * maxGradRef
                                    / (mean_grads_reg[i] + np.float64(1e-16)),
                                    a_min=np.float64(1e-1),
                                    a_max=np.float64(1e6),
                                )

                    else:
                        time_s = time.time()
                        loss_info = self.train_step(
                            x_batch_trainList=x_batch_trainList,
                            x_cs_batch_trainList=x_cs_batch_trainList,
                            x_params_batch_trainList=x_params_batch_trainList,
                            y_batch_trainList=y_batch_trainList,
                            int_col_pts=int_col_pts,
                            int_col_params=int_col_params,
                            bound_col_pts=bound_col_pts,
                            bound_col_params=bound_col_params,
                            reg_col_pts=reg_col_pts,
                            reg_col_params=reg_col_params,
                            tmax=new_tmax,
                            gradient_threshold=gradient_threshold,
                        )
                        grads_model = loss_info[9]

                    self.applyGrad(grads_model)
                    time_e = time.time()

                    mse = loss_info[0]
                    if self.dynamicAttentionWeights:
                        mse_unweighted = loss_info[14]
                    (
                        frac_intLoss,
                        frac_boundLoss,
                        frac_dataLoss,
                        frac_regLoss,
                        intTerms,
                        boundTerms,
                        dataTerms,
                        regTerms,
                    ) = loss_info[1:9]

                    if not optimized:
                        self.check_loss_dim(
                            intTerms,
                            boundTerms,
                            dataTerms,
                            regTerms,
                        )

                    lr_m, lr_m_old = self.dynamic_control_lrm(
                        lr_m, epoch, lrSchedulerModel
                    )
                    self.k_set_value(self.dsOptimizerModel.learning_rate, lr_m)
                    if self.dynamicAttentionWeights:
                        lr_w, lr_w_old = self.dynamic_control_lrw(
                            lr_w, epoch, lrSchedulerWeights
                        )
                        if self.activeInt:
                            self.k_set_value(
                                self.dsOptimizerInt.learning_rate, lr_w
                            )
                        if self.activeBound:
                            self.k_set_value(
                                self.dsOptimizerBound.learning_rate, lr_w
                            )
                        if self.activeData:
                            self.k_set_value(
                                self.dsOptimizerData.learning_rate, lr_w
                            )
                        if self.activeReg:
                            self.k_set_value(
                                self.dsOptimizerReg.learning_rate, lr_w
                            )
                    if self.useLossThreshold:
                        lt, lt_old = self.dynamic_control_lt(lt)

                    train_mse_loss = (
                        (step) * train_mse_loss + tf.reduce_sum(mse)
                    ) / (step + 1)
                    if self.dynamicAttentionWeights:
                        train_mse_unweighted_loss = (
                            (step) * train_mse_unweighted_loss
                            + tf.reduce_sum(mse_unweighted)
                        ) / (step + 1)

                    time_per_step = (
                        (step) * (time_per_step) + (time_e - time_s)
                    ) / (step + 1)

                    if not optimized and step % self.freq == 0:
                        print_progress_bar(
                            step,
                            self.n_batch,
                            prefix="Loss=%.2f i=%.2f b=%.2f d=%.2f r=%.2f, t/step=%.2g ms,  Epoch= %d / %d "
                            % (
                                train_mse_loss,
                                frac_intLoss,
                                frac_boundLoss,
                                frac_dataLoss,
                                frac_regLoss,
                                1e3 * time_per_step,
                                epoch + 1,
                                self.nEpochs,
                            ),
                            suffix="Complete",
                            length=20,
                        )
                    self.logLosses(
                        step + true_epoch_num * self.n_batch,
                        intTerms,
                        boundTerms,
                        dataTerms,
                        regTerms,
                    )
                if (
                    self.dynamicAttentionWeights
                    and epoch >= start_weight_training_epoch
                ):
                    if not self.printed_start_SA:
                        self.printed_start_SA = True
                        print("\nINFO: Start updating collocation weights\n")
                    if self.activeInt:
                        self.applyGrad_dynamicAttention_int(
                            tot_grad_int,
                            self.int_col_weights,
                        )
                    if self.activeBound:
                        self.applyGrad_dynamicAttention_bound(
                            tot_grad_bound,
                            self.bound_col_weights,
                        )
                    if self.activeData:
                        self.applyGrad_dynamicAttention_data(
                            tot_grad_data,
                            self.data_col_weights,
                        )
                    if self.activeReg:
                        self.applyGrad_dynamicAttention_reg(
                            tot_grad_reg,
                            self.reg_col_weights,
                        )

                print_progress_bar(
                    epoch + 1,
                    self.nEpochs,
                    prefix="Loss=%.2f i=%.2f b=%.2f d=%.2f r=%.2f, t/step=%.2g ms,  Epoch= %d / %d "
                    % (
                        train_mse_loss,
                        frac_intLoss,
                        frac_boundLoss,
                        frac_dataLoss,
                        frac_regLoss,
                        1e3 * time_per_step,
                        epoch + 1,
                        self.nEpochs,
                    ),
                    suffix="Complete",
                    length=20,
                )

                if self.dynamicAttentionWeights:
                    bestLoss = self.logTraining(
                        true_epoch_num,
                        mse=train_mse_loss,
                        bestLoss=bestLoss,
                        mse_unweighted=train_mse_unweighted_loss,
                    )
                else:
                    bestLoss = self.logTraining(
                        true_epoch_num, mse=train_mse_loss, bestLoss=bestLoss
                    )

                if currentEpoch_it == 1:
                    if self.dynamicAttentionWeights:
                        old_mse_unweighted_loss = train_mse_unweighted_loss
                        max_mse_unweighted_loss = train_mse_unweighted_loss
                        min_mse_unweighted_loss = train_mse_unweighted_loss
                    old_mse_loss = train_mse_loss
                    max_mse_loss = train_mse_loss
                    min_mse_loss = train_mse_loss

                elif currentEpoch_it > 1:
                    if self.dynamicAttentionWeights:
                        if train_mse_unweighted_loss > max_mse_unweighted_loss:
                            max_mse_unweighted_loss = train_mse_unweighted_loss
                        if train_mse_unweighted_loss < min_mse_unweighted_loss:
                            min_mse_unweighted_loss = train_mse_unweighted_loss
                    if train_mse_loss > max_mse_loss:
                        max_mse_loss = train_mse_loss
                    if train_mse_loss < min_mse_loss:
                        min_mse_loss = train_mse_loss

                safe_save(
                    self.model,
                    os.path.join(self.modelFolder, "lastSGD.weights.h5"),
                    overwrite=True,
                )
                safe_save(
                    self.model,
                    os.path.join(self.modelFolder, "last.weights.h5"),
                    overwrite=True,
                )
                if (true_epoch_num * self.n_batch) % 1000 == 0:
                    safe_save(
                        self.model,
                        os.path.join(
                            self.modelFolder,
                            f"step_{true_epoch_num * self.n_batch}.weights.h5",
                        ),
                        overwrite=True,
                    )
                    print("\nSaved weights")

                true_epoch_num += 1

                if self.dynamicAttentionWeights:
                    if train_mse_unweighted_loss > min_mse_unweighted_loss:
                        loss_increase_occurence += 1
                elif train_mse_loss > min_mse_loss:
                    loss_increase_occurence += 1

                if self.useLossThreshold:
                    if self.gradualTime_sgd and epoch < self.nEpochs // 2:
                        if (
                            loss_increase_occurence >= 3
                            or currentEpoch_it
                            >= min(inner_epochs, self.nEpochs // 2)
                        ):
                            loss_increase_occurence = 0
                            currentEpoch_it = 0
                            break
                        if self.dynamicAttentionWeights:
                            if train_mse_unweighted_loss < lt:
                                loss_increase_occurence = 0
                                currentEpoch_it = 0
                                break
                        else:
                            if train_mse_loss < lt:
                                loss_increase_occurence = 0
                                currentEpoch_it = 0
                                break
                    else:
                        if self.dynamicAttentionWeights:
                            if (
                                currentEpoch_it
                                == min(inner_epochs, self.nEpochs // 2)
                                and (
                                    train_mse_unweighted_loss
                                    / old_mse_unweighted_loss
                                )
                                > 1.0
                                and (train_mse_loss / old_mse_loss) > 1.0
                            ):
                                self.force_decrease_lrm_lrw(
                                    lr_m,
                                    learningRateModelFinal,
                                    lr_w,
                                    learningRateWeightsFinal,
                                )
                            if (
                                train_mse_unweighted_loss < lt
                                or currentEpoch_it
                                >= min(inner_epochs, self.nEpochs // 2)
                            ):
                                old_mse_unweighted_loss = (
                                    train_mse_unweighted_loss
                                )
                                old_mse_loss = train_mse_loss
                                loss_increase_occurence = 0
                                currentEpoch_it = 0
                                break
                        else:
                            if (
                                currentEpoch_it
                                == min(inner_epochs, self.nEpochs // 2)
                                and train_mse_loss / old_mse_loss > 1.0
                            ):
                                self.force_decrease_lrm(
                                    lr_m, learningRateModelFinal
                                )
                            if train_mse_loss < lt or currentEpoch_it >= min(
                                inner_epochs, self.nEpochs // 2
                            ):
                                old_mse_loss = train_mse_loss
                                loss_increase_occurence = 0
                                currentEpoch_it = 0
                                break
                else:
                    loss_increase_occurence = 0
                    currentEpoch_it = 0
                    break

        if self.lbfgs:
            print("\nStarting L-BFGS training")
            if self.dynamicAttentionWeights:
                self.flatIntColOnes = [
                    tf.ones((self.n_int, 1), dtype=tf.dtypes.float64)
                ] * len(self.interiorTerms_rescale)
                if self.activeInt:
                    self.flatIntColWeights = [
                        tf.reshape(
                            tf.convert_to_tensor(
                                np.array(
                                    [
                                        int_col_weights[i].numpy()
                                        for int_col_weights in self.int_col_weights
                                    ]
                                )
                            ),
                            (self.n_int, -1),
                        )
                        for i in range(len(self.interiorTerms_rescale))
                    ]
                else:
                    self.flatIntColWeights = [
                        tf.zeros((self.n_batch), dtype=tf.dtypes.float64)
                    ]
                self.flatBoundColOnes = [
                    tf.ones((self.n_bound, 1), dtype=tf.dtypes.float64)
                ] * len(self.boundaryTerms_rescale)
                if self.activeBound:
                    self.flatBoundColWeights = [
                        tf.reshape(
                            tf.convert_to_tensor(
                                np.array(
                                    [
                                        bound_col_weights[i].numpy()
                                        for bound_col_weights in self.bound_col_weights
                                    ]
                                )
                            ),
                            (self.n_bound, -1),
                        )
                        for i in range(len(self.boundaryTerms_rescale))
                    ]
                else:
                    self.flatBoundColWeights = [
                        tf.zeros((self.n_batch), dtype=tf.dtypes.float64)
                    ]
                self.flatDataColOnes = [
                    tf.ones((self.n_data, 1), dtype=tf.dtypes.float64)
                ] * len(self.dataTerms_rescale)
                if self.activeData:
                    self.flatDataColWeights = [
                        tf.reshape(
                            tf.convert_to_tensor(
                                np.array(
                                    [
                                        data_col_weights[i].numpy()
                                        for data_col_weights in self.data_col_weights
                                    ]
                                )
                            ),
                            (self.n_data, -1),
                        )
                        for i in range(len(self.dataTerms_rescale))
                    ]
                else:
                    self.flatDataColWeights = [
                        tf.zeros((self.n_batch), dtype=tf.dtypes.float64)
                    ]
                self.flatRegColOnes = [
                    tf.ones((self.n_reg, 1), dtype=tf.dtypes.float64)
                ] * len(self.regTerms_rescale)
                if self.activeReg:
                    self.flatRegColWeights = [
                        tf.reshape(
                            tf.convert_to_tensor(
                                np.array(
                                    [
                                        reg_col_weights[i].numpy()
                                        for reg_col_weights in self.reg_col_weights
                                    ]
                                )
                            ),
                            (self.n_reg, -1),
                        )
                        for i in range(len(self.regTerms_rescale))
                    ]
                else:
                    self.flatRegColWeights = [
                        tf.zeros((self.n_batch), dtype=tf.dtypes.float64)
                    ]

            if self.gradualTime_sgd or self.gradualTime_lbfgs:
                tmax = self.params["tmax"]
            else:
                tmax = None

            changeSetUp = False
            if self.collocationMode.lower() == "random":
                changeSetUp = True
                self.collocationMode = "fixed"
            self.run_SGD = False
            self.run_LBFGS = True

            nRemainingEp_lbfgs = self.nEpochs_lbfgs
            nDoneEp_lbfgs = 0
            if self.gradualTime_lbfgs:
                for istep_lbfgs in range(self.n_gradual_steps_lbfgs):
                    new_tmax = self.gradualTimeSchedule_lbfgs[istep_lbfgs]
                    print("")
                    print(f"GRAD {istep_lbfgs+1}/{self.n_gradual_steps_lbfgs}")
                    print("tmax = ", new_tmax)
                    print("")
                    bestLoss = self.runLBFGS(
                        tmax=new_tmax,
                        nIter=self.nEpochs_start_lbfgs
                        + self.nEp_per_gradual_step,
                        epochDoneLBFGS=nDoneEp_lbfgs,
                        epochDoneSGD=true_epoch_num,
                        bestLoss=bestLoss,
                        learningRateLBFGS=learningRateLBFGS,
                        gradient_threshold=gradient_threshold,
                    )
                    nRemainingEp_lbfgs -= self.nEp_per_gradual_step
                    nDoneEp_lbfgs += (
                        self.nEpochs_start_lbfgs + self.nEp_per_gradual_step
                    )

            print("")

            bestLoss = self.runLBFGS(
                tmax=tmax,
                nIter=nRemainingEp_lbfgs,
                epochDoneLBFGS=nDoneEp_lbfgs,
                epochDoneSGD=true_epoch_num,
                bestLoss=bestLoss,
                learningRateLBFGS=learningRateLBFGS,
                gradient_threshold=gradient_threshold,
            )

            if changeSetUp:
                self.collocationMode = "random"

            safe_save(
                self.model,
                os.path.join(self.modelFolder, "last.weights.h5"),
                overwrite=True,
            )

        if self.gradualTime_sgd or self.gradualTime_lbfgs:
            tmax = self.params["tmax"]
        else:
            tmax = None
        unweighted_loss = self.get_unweighted_loss(
            self.int_col_pts,
            self.int_col_params,
            self.bound_col_pts,
            self.bound_col_params,
            self.reg_col_pts,
            self.reg_col_params,
            [tf.convert_to_tensor(x) for x in self.xDataList_full],
            [tf.convert_to_tensor(x) for x in self.x_params_dataList_full],
            [tf.convert_to_tensor(y) for y in self.yDataList_full],
            self.n_batch_lbfgs,
            tmax=tmax,
        )

        return unweighted_loss

    def prepareLog(self):
        os.makedirs(self.modelFolder, exist_ok=True)
        os.makedirs(self.logLossFolder, exist_ok=True)
        try:
            os.remove(os.path.join(self.modelFolder, "config.json"))
        except:
            pass
        try:
            os.remove(os.path.join(self.logLossFolder, "log.csv"))
        except:
            pass
        try:
            os.remove(os.path.join(self.logLossFolder, "interiorTerms.csv"))
        except:
            pass
        try:
            os.remove(os.path.join(self.logLossFolder, "boundaryTerms.csv"))
        except:
            pass
        try:
            os.remove(os.path.join(self.logLossFolder, "dataTerms.csv"))
        except:
            pass
        try:
            os.remove(os.path.join(self.logLossFolder, "regTerms.csv"))
        except:
            pass
        if self.annealingWeights:
            try:
                os.remove(
                    os.path.join(self.logLossFolder, "int_loss_weights.csv")
                )
            except:
                pass
            try:
                os.remove(
                    os.path.join(self.logLossFolder, "bound_loss_weights.csv")
                )
            except:
                pass
            try:
                os.remove(
                    os.path.join(self.logLossFolder, "data_loss_weights.csv")
                )
            except:
                pass
            try:
                os.remove(
                    os.path.join(self.logLossFolder, "reg_loss_weights.csv")
                )
            except:
                pass

        # Save model configuration
        with open(
            os.path.join(self.modelFolder, "config.json"), "w+"
        ) as outfile:
            # float 32 is not supported by Json
            for key in self.config:
                ent_type = str(type(self.config[key]))
                if "numpy.float" in ent_type and "32" in ent_type:
                    self.config[key] = float(self.config[key])
                elif key == "params_min" or key == "params_max":
                    for ientry, entry in enumerate(self.config[key]):
                        ent_type = str(type(self.config[key][ientry]))
                        if "numpy.float" in ent_type and "32" in ent_type:
                            self.config[key][ientry] = float(entry)

            json.dump(self.config, outfile, indent=4, sort_keys=True)

        # Make log headers
        f = open(os.path.join(self.logLossFolder, "log.csv"), "a+")
        f.write("epoch;step;mseloss\n")
        f.close()
        f = open(os.path.join(self.logLossFolder, "interiorTerms.csv"), "a+")
        f.write("step;lossArray\n")
        f.close()
        f = open(os.path.join(self.logLossFolder, "boundaryTerms.csv"), "a+")
        f.write("step;lossArray\n")
        f.close()
        f = open(os.path.join(self.logLossFolder, "dataTerms.csv"), "a+")
        f.write("step;lossArray\n")
        f.close()
        f = open(os.path.join(self.logLossFolder, "regTerms.csv"), "a+")
        f.write("step;lossArray\n")
        f.close()
        if self.annealingWeights:
            f = open(
                os.path.join(self.logLossFolder, "int_loss_weights.csv"), "a+"
            )
            f.write("step;weightArray\n")
            f.close()
            f = open(
                os.path.join(self.logLossFolder, "bound_loss_weights.csv"),
                "a+",
            )
            f.write("step;weightArray\n")
            f.close()
            f = open(
                os.path.join(self.logLossFolder, "data_loss_weights.csv"), "a+"
            )
            f.write("step;weightArray\n")
            f.close()
            f = open(
                os.path.join(self.logLossFolder, "reg_loss_weights.csv"), "a+"
            )
            f.write("step;weightArray\n")
            f.close()

    def initLearningRateControl(
        self, learningRateModel, learningRateWeights=None
    ):
        f = open(os.path.join(self.modelFolder, "learningRateModel"), "w+")
        f.write(str(learningRateModel))
        f.close()
        self.learningRateModelStamp = os.stat(
            os.path.join(self.modelFolder, "learningRateModel")
        ).st_mtime
        if self.dynamicAttentionWeights:
            f = open(
                os.path.join(self.modelFolder, "learningRateWeights"), "w+"
            )
            f.write(str(learningRateWeights))
            f.close()
            self.learningRateWeightsStamp = os.stat(
                os.path.join(self.modelFolder, "learningRateWeights")
            ).st_mtime

    def initLossThresholdControl(self, lossThreshold):
        f = open(os.path.join(self.modelFolder, "lossThreshold"), "w+")
        f.write(str(lossThreshold))
        f.close()
        self.lossThresholdStamp = os.stat(
            os.path.join(self.modelFolder, "lossThreshold")
        ).st_mtime

    def update_param_file(self, param, param_old, mode):
        if mode == "lr_m":
            filename = os.path.join(self.modelFolder, "learningRateModel")
        elif mode == "lr_w":
            filename = os.path.join(self.modelFolder, "learningRateWeights")
        elif mode == "lt":
            filename = os.path.join(self.modelFolder, "lossThreshold")
        # Update control file
        if abs(param - param_old) > 1e-15:
            f = open(filename, "w+")
            try:
                f.write(str(param.numpy()))
            except:
                f.write(str(param))
            f.close()
            stamp = os.stat(filename).st_mtime
            if mode == "lr_m":
                self.learningRateModelStamp = stamp
            elif mode == "lr_w":
                self.learningRateWeightsStamp = stamp
            elif mode == "lt":
                self.lossThresholdStamp = stamp

    def read_param_file(self, param, param_old, mode):
        if mode == "lr_m":
            has_changed = self.lrmHasChanged()
            filename = os.path.join(self.modelFolder, "learningRateModel")
            prefix_info = "INFO: LR Model changed from"
            prefix_warning = "WARNING: LR Model has changed but could not be updated. Using LR"
        elif mode == "lr_w":
            has_changed = self.lrwHasChanged()
            filename = os.path.join(self.modelFolder, "learningRateWeights")
            prefix_info = "INFO: LR Weights changed from"
            prefix_warning = "WARNING: LR Weights has changed but could not be updated. Using LR"
        elif mode == "lt":
            has_changed = self.ltHasChanged()
            filename = os.path.join(self.modelFolder, "lossThreshold")
            prefix_info = "INFO: LT changed from"
            prefix_warning = (
                "WARNING: LT has changed but could not be updated. Using LT"
            )
        if has_changed:
            param_old = param
            try:
                f = open(filename, "r")
                lines = f.readlines()
                param = np.float64(lines[0])
                if mode == "lr_m":
                    self.lr_m_epoch_start = param
                if mode == "lr_w":
                    self.lr_w_epoch_start = param
                if mode == "lt":
                    self.lt_epoch_start = param
                f.close()
                if abs(param_old - param) > 1e-12:
                    print(
                        "\n"
                        + prefix_info
                        + " %.3g to %.3g\n" % (param_old, param)
                    )
            except:
                print("\n" + prefix_warning + " = %.3g\n" % param_old)
                pass
            stamp = os.stat(filename).st_mtime
            if mode == "lr_m":
                self.learningRateModelStamp = stamp
            elif mode == "lr_w":
                self.learningRateWeightsStamp = stamp
            elif mode == "lt":
                self.lossThresholdStamp = stamp
        return param, param_old

    def force_decrease_lrm(self, lr_m, learningRateModelFinal):
        lr_m_old = lr_m
        lr_m = max(lr_m * 0.5, learningRateModelFinal)
        filename = os.path.join(self.modelFolder, "learningRateModel")
        f = open(filename, "w+")
        try:
            f.write(str(lr_m.numpy()))
        except:
            f.write(str(lr_m))
        f.close()
        return

    def force_decrease_lrm_lrw(
        self, lr_m, learningRateModelFinal, lr_w, learningRateWeightsFinal
    ):
        self.force_decrease_lrm(lr_m, learningRateModelFinal)
        lr_w_old = lr_w
        lr_w = max(lr_w * 0.5, learningRateWeightsFinal)
        filename = os.path.join(self.modelFolder, "learningRateWeights")
        f = open(filename, "w+")
        try:
            f.write(str(lr_w.numpy()))
        except:
            f.write(str(lr_w))
        f.close()
        return

    def dynamic_control_lrm(self, lr_m, epoch, scheduler):
        lr_m_old = lr_m
        lr_m = scheduler(epoch, self.lr_m_epoch_start)
        if (self.total_step % self.freq) == 0:
            lr_m, lr_m_old = self.read_param_file(lr_m, lr_m_old, mode="lr_m")
            self.update_param_file(lr_m, lr_m_old, mode="lr_m")
        return lr_m, lr_m_old

    def dynamic_control_lrw(self, lr_w, epoch, scheduler):
        lr_w_old = lr_w
        lr_w = scheduler(epoch, self.lr_w_epoch_start)
        if (self.total_step % self.freq) == 0:
            lr_w, lr_w_old = self.read_param_file(lr_w, lr_w_old, mode="lr_w")
            self.update_param_file(lr_w, lr_w_old, mode="lr_w")
        return lr_w, lr_w_old

    def dynamic_control_lt(self, lt):
        lt_old = lt
        if (self.total_step % self.freq) == 0:
            lt, lt_old = self.read_param_file(lt, lt_old, mode="lt")
            self.update_param_file(lt, lt, mode="lt")
        return lt, lt_old

    def lrmHasChanged(self):
        change = abs(
            os.stat(
                os.path.join(self.modelFolder, "learningRateModel")
            ).st_mtime
            - self.learningRateModelStamp
        )
        return change > 1e-6

    def lrwHasChanged(self):
        change = abs(
            os.stat(
                os.path.join(self.modelFolder, "learningRateWeights")
            ).st_mtime
            - self.learningRateWeightsStamp
        )
        return change > 1e-6

    def ltHasChanged(self):
        change = abs(
            os.stat(os.path.join(self.modelFolder, "lossThreshold")).st_mtime
            - self.lossThresholdStamp
        )
        return change > 1e-6

    def k_set_value(self, x, value):
        value = np.asarray(value, dtype=x.dtype)
        x.assign(value)

    def logTraining(self, epoch, mse, bestLoss, mse_unweighted=None):
        f = open(os.path.join(self.logLossFolder, "log.csv"), "a+")
        if self.dynamicAttentionWeights:
            f.write(
                str(int(epoch))
                + ";"
                + str(int(epoch * self.n_batch))
                + ";"
                + str(mse.numpy())
                + ";"
                + str(mse_unweighted.numpy())
                + "\n"
            )
        else:
            f.write(
                str(int(epoch))
                + ";"
                + str(int(epoch * self.n_batch))
                + ";"
                + str(mse.numpy())
                + "\n"
            )
        f.close()

        if mse_unweighted is None:
            epochLoss = mse
        else:
            epochLoss = mse_unweighted

        # Save model weights
        safe_save(
            self.model,
            os.path.join(self.modelFolder, "lastSGD.weights.h5"),
            overwrite=True,
        )

        if self.collocationMode.lower() == "fixed":
            if self.dynamicAttentionWeights:
                # Save attention weights
                if self.activeInt:
                    np.save(
                        os.path.join(self.modelFolder, "sa_int_weights.npy"),
                        [
                            np.array(
                                [
                                    int_col_weights[i].numpy()
                                    for int_col_weights in self.int_col_weights
                                ]
                            ).flatten()
                            for i in range(len(self.interiorTerms_rescale))
                        ],
                        allow_pickle=True,
                    )
                if self.activeBound:
                    np.save(
                        os.path.join(self.modelFolder, "sa_bound_weights.npy"),
                        [
                            np.array(
                                [
                                    bound_col_weights[i].numpy()
                                    for bound_col_weights in self.bound_col_weights
                                ]
                            ).flatten()
                            for i in range(len(self.boundaryTerms_rescale))
                        ],
                        allow_pickle=True,
                    )
                if self.activeData:
                    np.save(
                        os.path.join(self.modelFolder, "sa_data_weights.npy"),
                        [
                            np.array(
                                [
                                    data_col_weights[i].numpy()
                                    for data_col_weights in self.data_col_weights
                                ]
                            ).flatten()
                            for i in range(len(self.dataTerms_rescale))
                        ],
                        allow_pickle=True,
                    )
                if self.activeReg:
                    np.save(
                        os.path.join(self.modelFolder, "sa_reg_weights.npy"),
                        [
                            np.array(
                                [
                                    reg_col_weights[i].numpy()
                                    for reg_col_weights in self.reg_col_weights
                                ]
                            ).flatten()
                            for i in range(len(self.regTerms_rescale))
                        ],
                        allow_pickle=True,
                    )
            np.save(
                os.path.join(self.modelFolder, "t_data_col.npy"),
                [
                    np.array(self.xDataList_full[i][:, self.ind_t])
                    for i in range(len(self.dataTerms_rescale))
                ],
                allow_pickle=True,
            )
            np.save(
                os.path.join(self.modelFolder, "r_data_col.npy"),
                [
                    np.array(self.xDataList_full[i][:, self.ind_r])
                    for i in self.csDataTerms_ind
                ],
                allow_pickle=True,
            )
            if self.gradualTime_sgd:
                np.save(
                    os.path.join(self.modelFolder, "t_int_col.npy"),
                    self.stretchT(
                        self.int_col_pts[self.ind_int_col_t],
                        self.tmin_int_bound,
                        self.firstTime,
                        self.tmin_int_bound,
                        self.params["tmax"],
                    ),
                )
                np.save(
                    os.path.join(self.modelFolder, "t_bound_col.npy"),
                    self.stretchT(
                        self.bound_col_pts[self.ind_bound_col_t],
                        self.tmin_int_bound,
                        self.firstTime,
                        self.tmin_int_bound,
                        self.params["tmax"],
                    ),
                )
                np.save(
                    os.path.join(self.modelFolder, "t_reg_col.npy"),
                    self.stretchT(
                        self.reg_col_pts[self.ind_reg_col_t],
                        self.tmin_int_bound,
                        self.firstTime,
                        self.tmin_int_bound,
                        self.params["tmax"],
                    ),
                )
            else:
                np.save(
                    os.path.join(self.modelFolder, "t_int_col.npy"),
                    np.array(self.int_col_pts[self.ind_int_col_t]),
                )
                np.save(
                    os.path.join(self.modelFolder, "t_bound_col.npy"),
                    np.array(self.bound_col_pts[self.ind_bound_col_t]),
                )
                np.save(
                    os.path.join(self.modelFolder, "t_reg_col.npy"),
                    np.array(self.reg_col_pts[self.ind_reg_col_t]),
                )

            if self.activeInt:
                np.save(
                    os.path.join(self.modelFolder, "r_a_int_col.npy"),
                    np.array(self.int_col_pts[self.ind_int_col_r_a]),
                )
                np.save(
                    os.path.join(self.modelFolder, "r_c_int_col.npy"),
                    np.array(self.int_col_pts[self.ind_int_col_r_c]),
                )
                np.save(
                    os.path.join(self.modelFolder, "r_maxa_int_col.npy"),
                    np.array(self.int_col_pts[self.ind_int_col_r_maxa]),
                )
                np.save(
                    os.path.join(self.modelFolder, "r_maxc_int_col.npy"),
                    np.array(self.int_col_pts[self.ind_int_col_r_maxc]),
                )
                np.save(
                    os.path.join(self.modelFolder, "deg_i0_a_int_col.npy"),
                    np.array(
                        self.int_col_params[self.ind_int_col_params_deg_i0_a]
                    ),
                )
                np.save(
                    os.path.join(self.modelFolder, "deg_ds_c_int_col.npy"),
                    np.array(
                        self.int_col_params[self.ind_int_col_params_deg_ds_c]
                    ),
                )

            if self.activeBound:
                np.save(
                    os.path.join(self.modelFolder, "r_min_bound_col.npy"),
                    np.array(self.bound_col_pts[self.ind_bound_col_r_min]),
                )
                np.save(
                    os.path.join(self.modelFolder, "r_maxa_bound_col.npy"),
                    np.array(self.bound_col_pts[self.ind_bound_col_r_maxa]),
                )
                np.save(
                    os.path.join(self.modelFolder, "r_maxc_bound_col.npy"),
                    np.array(self.bound_col_pts[self.ind_bound_col_r_maxc]),
                )
                np.save(
                    os.path.join(self.modelFolder, "deg_i0_a_bound_col.npy"),
                    np.array(
                        self.bound_col_params[
                            self.ind_bound_col_params_deg_i0_a
                        ]
                    ),
                )
                np.save(
                    os.path.join(self.modelFolder, "deg_ds_c_bound_col.npy"),
                    np.array(
                        self.bound_col_params[
                            self.ind_bound_col_params_deg_ds_c
                        ]
                    ),
                )

            if self.activeReg:
                pass

        if bestLoss is None or epochLoss < bestLoss:
            bestLoss = epochLoss
            safe_save(
                self.model,
                os.path.join(self.modelFolder, "best.weights.h5"),
                overwrite=True,
            )
        return bestLoss

    def logLosses(
        self, step, interiorTerms, boundaryTerms, dataTerms, regTerms
    ):
        if step % self.freq == 0:
            if self.activeInt:
                interiorTermsArray = [
                    np.mean(tf.square(interiorTerm))
                    for interiorTerm in interiorTerms
                ]
            else:
                interiorTermsArray = []
            interiorTermsArrayPercent = [
                round(term / (1e-16 + sum(interiorTermsArray)), 2)
                for term in interiorTermsArray
            ]
            f = open(
                os.path.join(self.logLossFolder, "interiorTerms.csv"), "a+"
            )
            f.write(
                str(int(step)) + ";" + str(interiorTermsArrayPercent) + "\n"
            )
            f.write(str(int(step)) + ";" + str(interiorTermsArray) + "\n")
            f.close()
            if self.activeBound:
                boundaryTermsArray = [
                    np.mean(tf.square(boundaryTerm))
                    for boundaryTerm in boundaryTerms
                ]
            else:
                boundaryTermsArray = []
            boundaryTermsArrayPercent = [
                round(term / (1e-16 + sum(boundaryTermsArray)), 2)
                for term in boundaryTermsArray
            ]
            f = open(
                os.path.join(self.logLossFolder, "boundaryTerms.csv"), "a+"
            )
            f.write(
                str(int(step)) + ";" + str(boundaryTermsArrayPercent) + "\n"
            )
            f.write(str(int(step)) + ";" + str(boundaryTermsArray) + "\n")
            f.close()
            if self.activeData:
                dataTermsArray = [
                    np.mean(tf.square(dataTerm)) for dataTerm in dataTerms
                ]
            else:
                dataTermsArray = []
            dataTermsArrayPercent = [
                round(term / (1e-16 + sum(dataTermsArray)), 2)
                for term in dataTermsArray
            ]
            f = open(os.path.join(self.logLossFolder, "dataTerms.csv"), "a+")
            f.write(str(int(step)) + ";" + str(dataTermsArrayPercent) + "\n")
            f.write(str(int(step)) + ";" + str(dataTermsArray) + "\n")
            f.close()
            if self.activeReg:
                regTermsArray = [
                    np.mean(tf.square(regTerm)) for regTerm in regTerms
                ]
            else:
                regTermsArray = []
            regTermsArrayPercent = [
                round(term / (1e-16 + sum(regTermsArray)), 2)
                for term in regTermsArray
            ]
            f = open(os.path.join(self.logLossFolder, "regTerms.csv"), "a+")
            f.write(str(int(step)) + ";" + str(regTermsArrayPercent) + "\n")
            f.write(str(int(step)) + ";" + str(regTermsArray) + "\n")
            f.close()

            if self.annealingWeights:
                if self.activeInt:
                    int_weights_array = []
                    for weight in self.int_loss_weights:
                        try:
                            int_weights_array.append(weight.numpy())
                        except:
                            int_weights_array.append(weight)
                else:
                    int_weights_array = []
                f = open(
                    os.path.join(self.logLossFolder, "int_loss_weights.csv"),
                    "a+",
                )
                f.write(str(int(step)) + ";" + str(int_weights_array) + "\n")
                f.close()
                if self.activeBound:
                    bound_weights_array = []
                    for weight in self.bound_loss_weights:
                        try:
                            bound_weights_array.append(weight.numpy())
                        except:
                            bound_weights_array.append(weight)
                else:
                    bound_weights_array = []
                f = open(
                    os.path.join(self.logLossFolder, "bound_loss_weights.csv"),
                    "a+",
                )
                f.write(str(int(step)) + ";" + str(bound_weights_array) + "\n")
                f.close()
                if self.activeData:
                    for weight in self.data_loss_weights:
                        try:
                            data_weights_array.append(weight.numpy())
                        except:
                            data_weights_array.append(weight)
                else:
                    data_weights_array = []
                f = open(
                    os.path.join(self.logLossFolder, "data_loss_weights.csv"),
                    "a+",
                )
                f.write(str(int(step)) + ";" + str(data_weights_array) + "\n")
                f.close()
                if self.activeReg:
                    for weight in self.reg_loss_weights:
                        try:
                            reg_weights_array.append(weight.numpy())
                        except:
                            reg_weights_array.append(weight)
                else:
                    reg_weights_array = []
                f = open(
                    os.path.join(self.logLossFolder, "reg_loss_weights.csv"),
                    "a+",
                )
                f.write(str(int(step)) + ";" + str(reg_weights_array) + "\n")
                f.close()

        return
