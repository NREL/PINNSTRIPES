import os
import sys
import time

import argument
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from _losses import (
    loss_fn,
    loss_fn_dynamicAttention_tensor,
    loss_fn_lbfgs,
    loss_fn_lbfgs_SA,
)
from conditionalDecorator import conditional_decorator
from custom_activations import swish_activation
from dataTools import checkDataShape, completeDataset
from eager_lbfgs import Struct, lbfgs
from progressBar import printProgressBar
from tensorflow.keras import backend as K
from tensorflow.keras import (
    initializers,
    layers,
    losses,
    optimizers,
    regularizers,
)
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.constraints import max_norm, unit_norm
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

tf.keras.backend.set_floatx("float64")

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
    return out


def resblock(x, n_units, n_layers, initializer, activation):
    tmp_x = x
    for ilayer in range(n_layers):
        tmp_x = Dense(n_units, kernel_initializer=initializer)(x)
        tmp_x = flexible_activation(tmp_x, activation)
    out = layers.Add()([x, tmp_x])
    out = flexible_activation(out, activation)

    return out


class pinn(Model):
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
        gradualTime=False,
        firstTime=np.float64(0.1),
        tmin_int_bound=np.float64(0.1),
        nEpochs=60,
        nEpochs_lbfgs=60,
        initialLossThreshold=np.float64(100),
        dynamicAttentionWeights=False,
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
    ):
        super(pinn, self).__init__()

        if optimized:
            self.freq = 100
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
        self.hidden_units_t = hidden_units_t
        self.hidden_units_t_r = hidden_units_t_r
        self.hidden_units_phie = hidden_units_phie
        self.hidden_units_phis_c = hidden_units_phis_c
        self.hidden_units_cs_a = hidden_units_cs_a
        self.hidden_units_cs_c = hidden_units_cs_c
        self.n_hidden_res_blocks = n_hidden_res_blocks
        self.n_res_block_layers = n_res_block_layers
        self.n_res_block_units = n_res_block_units
        self.dynamicAttentionWeights = dynamicAttentionWeights
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
        # self.phie0 = self.params["phie0"]
        self.phis_a0 = np.float64(0.0)
        # self.phis_c0 = self.params["phis_c0"]
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

        self.gradualTime = gradualTime

        # Self dynamics attention weights not allowed with random col
        if self.collocationMode.lower() == "random":
            if self.dynamicAttentionWeights:
                print(
                    "WARNING: dynamic attention weights not allowed with random collocation points"
                )
                print("WARNING: Disabling dynamic attention weights")
            self.dynamicAttentionWeights = False

        if self.gradualTime:
            self.firstTime = np.float64(firstTime)
            self.timeIncreaseExponent = -np.log(
                (self.firstTime - np.float64(self.params["tmin"]))
                / (
                    np.float64(self.params["tmax"])
                    - np.float64(self.params["tmin"])
                )
            )

        self.reg = 0  # 1e-3
        self.n_batch = n_batch
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
            print("new n data = ", self.new_nData)
            print("batch_size_data = ", self.batch_size_data)
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
            print("INFO: INT loss is INACTIVE")
            self.activeInt = False
            n_int = n_batch
            self.batch_size_int = 1
        else:
            print("INFO: INT loss is ACTIVE")
        if self.batch_size_bound == 0 or abs(self.alpha[1]) < 1e-12:
            print("INFO: BOUND loss is INACTIVE")
            self.activeBound = False
            n_bound = n_batch
            self.batch_size_bound = 1
        else:
            print("INFO: BOUND loss is ACTIVE")
        if (
            self.max_batch_size_data == 0
            or abs(self.alpha[2]) < 1e-12
            or xDataList == []
        ):
            print("INFO: DATA loss is INACTIVE")
            self.activeData = False
            n_data = n_batch
            self.batch_size_data = 1
        else:
            print("INFO: DATA loss is ACTIVE")
        if self.batch_size_reg == 0 or abs(self.alpha[3]) < 1e-12:
            print("INFO: REG loss is INACTIVE")
            self.activeReg = False
            n_reg = n_batch
            self.batch_size_reg = 1
        else:
            print("INFO: REG loss is ACTIVE")

        self.setResidualRescaling()

        # Interior loss collocation points
        self.r_a_int = tf.random.uniform(
            (n_int, 1),
            minval=self.rmin,
            maxval=self.rmax_a,
            dtype=tf.dtypes.float64,
        )
        self.r_c_int = tf.random.uniform(
            (n_int, 1),
            minval=self.rmin,
            maxval=self.rmax_c,
            dtype=tf.dtypes.float64,
        )
        self.r_maxa_int = self.rmax_a * tf.ones(
            (n_int, 1), dtype=tf.dtypes.float64
        )
        self.r_maxc_int = self.rmax_c * tf.ones(
            (n_int, 1), dtype=tf.dtypes.float64
        )
        if self.gradualTime:
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
        if self.gradualTime:
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
        if self.gradualTime:
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

        if not self.hidden_units_t is None:
            self.makeMergedModel()
        else:
            self.makeSplitModel()

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
                    "Adjust N_BATCH and MAX_BATCH_SIZE_DATA to accomodate %d datapoints"
                    % ndata_orig
                )
        else:
            self.new_nData = self.n_batch
            self.xDataList_full = [
                np.zeros((self.n_batch, self.dim_inpt)).astype("float64")
                if i in [self.ind_cs_a_data, self.ind_cs_c_data]
                else np.zeros((self.n_batch, self.dim_inpt - 1)).astype(
                    "float64"
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

        print("n_batch_sgd = ", self.n_batch)
        print("n_epoch_sgd = ", self.nEpochs)
        if self.lbfgs:
            self.nEpochs_lbfgs = nEpochs_lbfgs
            self.nEpochs_start_lbfgs = nEpochs_start_lbfgs
            # Do nEpochs_start_lbfgs iterations at first to make sure the Hessian is correctly computed
            self.nEpochs_lbfgs += self.nEpochs_start_lbfgs
            self.n_batch_lbfgs = n_batch_lbfgs
            print("n_batch_lbfgs = ", self.n_batch_lbfgs)
            print("n_epoch_lbfgs = ", self.nEpochs_lbfgs)
            if self.n_batch_lbfgs > self.n_batch:
                sys.exit(
                    "ERROR: n_batch LBFGS must be smaller or equal to SGD's"
                )
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
            if self.nEpochs_lbfgs <= self.nEpochs_start_lbfgs:
                self.lbfgs = False
                print(
                    "WARNING: Will not use LBFGS based on number of epoch specified"
                )
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
            # print("sizes w = ",self.sizes_w)
            # print("sizes b = ",self.sizes_b)
        self.sgd = sgd
        if self.nEpochs <= 0:
            self.sgd = False
            print(
                "WARNING: Will not use SGD based on number of epoch specified"
            )
            self.dynamicAttentionWeights = False

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
        self.config["hard_IC_timescale"] = self.hard_IC_timescale
        self.config["exponentialLimiter"] = self.exponentialLimiter
        self.config["dynamicAttentionWeights"] = self.dynamicAttentionWeights
        self.config["linearizeJ"] = self.linearizeJ
        self.config["activation"] = self.activation
        self.config["activeInt"] = self.activeInt
        self.config["activeBound"] = self.activeBound
        self.config["activeData"] = self.activeData
        self.config["activeReg"] = self.activeReg
        self.config["params_min"] = self.params_min
        self.config["params_max"] = self.params_max

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
                n_terms = len(self.regularizationTerms_rescale)
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

    def makeSplitModel(self):

        print("INFO: MAKING SPLIT MODEL")

        # inputs
        input_t = Input(shape=(1,), name="input_t")
        input_r = Input(shape=(1,), name="input_r")
        input_deg_i0_a = Input(shape=(1,), name="input_deg_i0_a")
        input_deg_ds_c = Input(shape=(1,), name="input_deg_ds_c")
        input_t_par = concatenate(
            [input_t, input_deg_i0_a, input_deg_ds_c], name="input_t_par"
        )

        initializer = "glorot_normal"

        # phie
        tmp_phie = input_t_par
        for unit in self.hidden_units_phie:
            tmp_phie = Dense(unit, kernel_initializer=initializer)(tmp_phie)
            tmp_phie = flexible_activation(tmp_phie, self.activation)
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_phie = resblock(
                tmp_phie,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_phie = Dense(1, activation="linear", name="phie")(tmp_phie)

        # phis_c
        tmp_phis_c = input_t_par
        for unit in self.hidden_units_phis_c:
            tmp_phis_c = Dense(unit, kernel_initializer=initializer)(
                tmp_phis_c
            )
            tmp_phis_c = flexible_activation(tmp_phis_c, self.activation)
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_phis_c = resblock(
                tmp_phis_c,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_phis_c = Dense(1, activation="linear", name="phis_c")(
            tmp_phis_c
        )

        # cs_a
        tmp_cs_a = concatenate([input_t_par, input_r], name="input_cs_a")
        for unit in self.hidden_units_cs_a:
            tmp_cs_a = Dense(unit, kernel_initializer=initializer)(tmp_cs_a)
            tmp_cs_a = flexible_activation(tmp_cs_a, self.activation)
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_cs_a = resblock(
                tmp_cs_a,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_cs_a = Dense(1, activation="sigmoid", name="cs_a")(tmp_cs_a)

        # cs_c
        tmp_cs_c = concatenate([input_t_par, input_r], name="input_cs_c")
        for unit in self.hidden_units_cs_c:
            tmp_cs_c = Dense(unit, kernel_initializer=initializer)(tmp_cs_c)
            tmp_cs_c = flexible_activation(tmp_cs_c, self.activation)
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_cs_c = resblock(
                tmp_cs_c,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_cs_c = Dense(1, activation="sigmoid", name="cs_c")(tmp_cs_c)

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
        print("INFO: MAKING MERGED MODEL")

        # inputs
        input_t = Input(shape=(1,), name="input_t")
        input_r = Input(shape=(1,), name="input_r")
        input_deg_i0_a = Input(shape=(1,), name="input_deg_i0_a")
        input_deg_ds_c = Input(shape=(1,), name="input_deg_ds_c")
        input_t_par = concatenate(
            [input_t, input_deg_i0_a, input_deg_ds_c], name="input_t_par"
        )

        initializer = "glorot_normal"

        # t domain
        tmp_t = input_t_par
        for unit in self.hidden_units_t:
            tmp_t = Dense(unit, kernel_initializer=initializer)(tmp_t)
            tmp_t = flexible_activation(tmp_t, self.activation)

        # t_r domain
        tmp_t_r = concatenate([tmp_t, input_r], name="input_t_r")
        for unit in self.hidden_units_t_r:
            tmp_t_r = Dense(unit, kernel_initializer=initializer)(tmp_t_r)
            tmp_t_r = flexible_activation(tmp_t_r, self.activation)

        # phie
        tmp_phie = tmp_t
        for unit in self.hidden_units_phie:
            tmp_phie = Dense(unit, kernel_initializer=initializer)(tmp_phie)
            tmp_phie = flexible_activation(tmp_phie, self.activation)
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_phie = resblock(
                tmp_phie,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_phie = Dense(1, activation="linear", name="phie")(tmp_phie)

        # phis_c
        tmp_phis_c = tmp_t
        for unit in self.hidden_units_phis_c:
            tmp_phis_c = Dense(unit, kernel_initializer=initializer)(
                tmp_phis_c
            )
            tmp_phis_c = flexible_activation(tmp_phis_c, self.activation)

        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_phis_c = resblock(
                tmp_phis_c,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_phis_c = Dense(1, activation="linear", name="phis_c")(
            tmp_phis_c
        )

        # cs_a
        tmp_cs_a = tmp_t_r
        for unit in self.hidden_units_cs_a:
            tmp_cs_a = Dense(unit, kernel_initializer=initializer)(tmp_cs_a)
            tmp_cs_a = flexible_activation(tmp_cs_a, self.activation)
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_cs_a = resblock(
                tmp_cs_a,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_cs_a = Dense(1, activation="sigmoid", name="cs_a")(tmp_cs_a)

        # cs_c
        tmp_cs_c = tmp_t_r
        for unit in self.hidden_units_cs_c:
            tmp_cs_c = Dense(unit, kernel_initializer=initializer)(tmp_cs_c)
            tmp_cs_c = flexible_activation(tmp_cs_c, self.activation)
        for i_res_block in range(self.n_hidden_res_blocks):
            tmp_cs_c = resblock(
                tmp_cs_c,
                self.n_res_block_units,
                self.n_res_block_layers,
                initializer,
                self.activation,
            )
        output_cs_c = Dense(1, activation="sigmoid", name="cs_c")(tmp_cs_c)

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
        if self.gradualTime:
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
            # print("layer ", ilayer)
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

    from _rescale import rescaleCs_a, rescaleCs_c, rescalePhie, rescalePhis_c

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
        get_loss_and_flat_grad_SA,
        interior_loss,
        regularization_loss,
        setResidualRescaling,
    )
    from dataTools import check_loss_dim

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
    ):

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
        printProgressBar(
            0,
            self.nEpochs,
            prefix="Loss=%s  Epoch= %d / %d " % ("?", 0, self.nEpochs),
            suffix="Complete",
            length=20,
        )
        true_epoch_num = 0
        self.printed_start_SA = False

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
            if self.gradualTime:
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
                    K.set_value(self.dsOptimizerModel.learning_rate, lr_m)
                    if self.dynamicAttentionWeights:
                        lr_w, lr_w_old = self.dynamic_control_lrw(
                            lr_w, epoch, lrSchedulerWeights
                        )
                        if self.activeInt:
                            K.set_value(
                                self.dsOptimizerInt.learning_rate, lr_w
                            )
                        if self.activeBound:
                            K.set_value(
                                self.dsOptimizerBound.learning_rate, lr_w
                            )
                        if self.activeData:
                            K.set_value(
                                self.dsOptimizerData.learning_rate, lr_w
                            )
                        if self.activeReg:
                            K.set_value(
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
                        printProgressBar(
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

                printProgressBar(
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

                self.model.save_weights(
                    os.path.join(self.modelFolder, "lastSGD.h5"),
                    overwrite=True,
                )
                self.model.save_weights(
                    os.path.join(self.modelFolder, "last.h5"), overwrite=True
                )

                true_epoch_num += 1

                if self.dynamicAttentionWeights:
                    if train_mse_unweighted_loss > min_mse_unweighted_loss:
                        loss_increase_occurence += 1
                elif train_mse_loss > min_mse_loss:
                    loss_increase_occurence += 1

                if self.useLossThreshold:
                    if self.gradualTime and epoch < self.nEpochs // 2:
                        if (
                            loss_increase_occurence >= 3
                            or currentEpoch_it
                            >= min(inner_epochs * 5, self.nEpochs // 2)
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
            print("Starting L-BFGS training")
            if self.dynamicAttentionWeights:
                if self.activeInt:
                    flatIntColWeights = [
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
                    flatIntColWeights = [
                        tf.zeros((self.n_batch), dtype=tf.dtypes.float64)
                    ]
                if self.activeBound:
                    flatBoundColWeights = [
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
                    flatBoundColWeights = [
                        tf.zeros((self.n_batch), dtype=tf.dtypes.float64)
                    ]
                if self.activeData:
                    flatDataColWeights = [
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
                    flatDataColWeights = [
                        tf.zeros((self.n_batch), dtype=tf.dtypes.float64)
                    ]
                if self.activeReg:
                    flatRegColWeights = [
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
                    flatRegColWeights = [
                        tf.zeros((self.n_batch), dtype=tf.dtypes.float64)
                    ]

            if self.gradualTime:
                tmax = self.params["tmax"]
            else:
                tmax = None

            changeSetUp = False
            if self.collocationMode.lower() == "random":
                changeSetUp = True
                self.collocationMode = "fixed"

            if self.dynamicAttentionWeights:
                loss_and_flat_grad = self.get_loss_and_flat_grad_SA(
                    self.int_col_pts,
                    self.int_col_params,
                    self.bound_col_pts,
                    self.bound_col_params,
                    self.reg_col_pts,
                    self.reg_col_params,
                    flatIntColWeights,
                    flatBoundColWeights,
                    flatDataColWeights,
                    flatRegColWeights,
                    [tf.convert_to_tensor(x) for x in self.xDataList_full],
                    [
                        tf.convert_to_tensor(x)
                        for x in self.x_params_dataList_full
                    ],
                    [tf.convert_to_tensor(y) for y in self.yDataList_full],
                    self.n_batch_lbfgs,
                    tmax=tmax,
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
                    [
                        tf.convert_to_tensor(x)
                        for x in self.x_params_dataList_full
                    ],
                    [tf.convert_to_tensor(y) for y in self.yDataList_full],
                    self.n_batch_lbfgs,
                    tmax=tmax,
                )

            lbfgs(
                loss_and_flat_grad,
                self.get_weights(self.model),
                Struct(),
                model=self.model,
                bestLoss=bestLoss,
                modelFolder=self.modelFolder,
                maxIter=self.nEpochs_lbfgs,
                learningRate=learningRateLBFGS,
                dynamicAttention=self.dynamicAttentionWeights,
                logLossFolder=self.logLossFolder,
                nEpochStart=true_epoch_num,
                nBatchSGD=self.n_batch,
                nEpochs_start_lbfgs=self.nEpochs_start_lbfgs,
            )

            if changeSetUp:
                self.collocationMode = "random"

            self.model.save_weights(
                os.path.join(self.modelFolder, "last.h5"), overwrite=True
            )

    def prepareLog(self):
        os.makedirs(self.modelFolder, exist_ok=True)
        os.makedirs(self.logLossFolder, exist_ok=True)
        try:
            os.remove(os.path.join(self.modelFolder, "config.npy"))
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

        # Save model configuration
        np.save(os.path.join(self.modelFolder, "config.npy"), self.config)

        # Make log headers
        f = open(os.path.join(self.logLossFolder, "log.csv"), "a+")
        f.write("epoch;mseloss\n")
        f.close()
        f = open(os.path.join(self.logLossFolder, "interiorTerms.csv"), "a+")
        f.write("epoch;lossArray\n")
        f.close()
        f = open(os.path.join(self.logLossFolder, "boundaryTerms.csv"), "a+")
        f.write("epoch;lossArray\n")
        f.close()
        f = open(os.path.join(self.logLossFolder, "dataTerms.csv"), "a+")
        f.write("epoch;lossArray\n")
        f.close()
        f = open(os.path.join(self.logLossFolder, "regTerms.csv"), "a+")
        f.write("epoch;lossArray\n")
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
        # if (self.total_step % self.freq) == 0 and has_changed:
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

    def logTraining(self, epoch, mse, bestLoss, mse_unweighted=None):

        f = open(os.path.join(self.logLossFolder, "log.csv"), "a+")
        if self.dynamicAttentionWeights:
            f.write(
                str(int(epoch))
                + ";"
                + str(mse.numpy())
                + ";"
                + str(mse_unweighted.numpy())
                + "\n"
            )
        else:
            f.write(str(int(epoch)) + ";" + str(mse.numpy()) + "\n")
        f.close()

        if mse_unweighted is None:
            epochLoss = mse
        else:
            epochLoss = mse_unweighted

        # Save model weights
        self.model.save_weights(
            os.path.join(self.modelFolder, "lastSGD.h5"), overwrite=True
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
            if self.gradualTime:
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
            self.model.save_weights(
                os.path.join(self.modelFolder, "best.h5"), overwrite=True
            )
            # tf.print("SAVING")
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

        return
