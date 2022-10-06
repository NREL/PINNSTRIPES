import os
import sys

import numpy as np

sys.path.append("util")
import shutil

import argument

# NN Stuff
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pinn import *
from pinn_parser import parseInputFile
from progressBar import printProgressBar
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import plot_model

tf.keras.backend.set_floatx("float64")

# Read command line arguments
args = argument.initArg()

from spm_simpler import *

params = makeParams()
dataFolder = args.dataFolder
inpt = parseInputFile(args.input_file)

seed = int(inpt["seed"])
tf.random.set_seed(seed)
np.random.seed(seed)

try:
    ID = int(inpt["ID"])
except KeyError:
    ID = 0

EPOCHS = int(inpt["EPOCHS"])
EPOCHS_LBFGS = int(inpt["EPOCHS_LBFGS"])
try:
    EPOCHS_START_LBFGS = int(inpt["EPOCHS_START_LBFGS"])
except KeyError:
    EPOCHS_START_LBFGS = 20
# Interior Bound Data Regularization
alpha = [float(entry) for entry in inpt["alpha"].split()]
LEARNING_RATE_WEIGHTS = float(inpt["LEARNING_RATE_WEIGHTS"])
LEARNING_RATE_WEIGHTS_FINAL = float(inpt["LEARNING_RATE_WEIGHTS_FINAL"])
LEARNING_RATE_MODEL = float(inpt["LEARNING_RATE_MODEL"])
LEARNING_RATE_MODEL_FINAL = float(inpt["LEARNING_RATE_MODEL_FINAL"])
LEARNING_RATE_LBFGS = float(inpt["LEARNING_RATE_LBFGS"])
HARD_IC_TIMESCALE = float(inpt["HARD_IC_TIMESCALE"])
RATIO_FIRST_TIME = float(inpt["RATIO_FIRST_TIME"])
RATIO_T_MIN = float(inpt["RATIO_T_MIN"])
EXP_LIMITER = float(inpt["EXP_LIMITER"])
COLLOCATION_MODE = inpt["COLLOCATION_MODE"]
# COLLOCATION_MODE = "random"
GRADUAL_TIME = inpt["GRADUAL_TIME"] == "True"
DYNAMIC_ATTENTION_WEIGHTS = inpt["DYNAMIC_ATTENTION_WEIGHTS"] == "True"
USE_LOSS_THRESHOLD = inpt["USE_LOSS_THRESHOLD"] == "True"
try:
    LOSS_THRESHOLD = float(inpt["LOSS_THRESHOLD"])
except KeyError:
    LOSS_THRESHOLD = 1000.0
try:
    INNER_EPOCHS = int(inpt["INNER_EPOCHS"])
except KeyError:
    INNER_EPOCHS = 1
try:
    START_WEIGHT_TRAINING_EPOCH = int(inpt["START_WEIGHT_TRAINING_EPOCH"])
except KeyError:
    START_WEIGHT_TRAINING_EPOCH = 1
ACTIVATION = inpt["ACTIVATION"]
LBFGS = inpt["LBFGS"] == "True"
SGD = inpt["SGD"] == "True"
MERGED = inpt["MERGED"] == "True"
LINEARIZE_J = inpt["LINEARIZE_J"] == "True"

# Surrogate NN
BATCH_SIZE_INT = int(inpt["BATCH_SIZE_INT"])
BATCH_SIZE_BOUND = int(inpt["BATCH_SIZE_BOUND"])
MAX_BATCH_SIZE_DATA = int(inpt["MAX_BATCH_SIZE_DATA"])
BATCH_SIZE_REG = int(inpt["BATCH_SIZE_REG"])
BATCH_SIZE_STRUCT = int(inpt["BATCH_SIZE_STRUCT"])
N_BATCH = int(inpt["N_BATCH"])
N_BATCH_LBFGS = int(inpt["N_BATCH_LBFGS"])
NEURONS_NUM = int(inpt["NEURONS_NUM"])
LAYERS_T_NUM = int(inpt["LAYERS_T_NUM"])
LAYERS_TR_NUM = int(inpt["LAYERS_TR_NUM"])
LAYERS_T_VAR_NUM = int(inpt["LAYERS_T_VAR_NUM"])
LAYERS_TR_VAR_NUM = int(inpt["LAYERS_TR_VAR_NUM"])
LAYERS_SPLIT_NUM = int(inpt["LAYERS_SPLIT_NUM"])
try:
    NUM_RES_BLOCKS = int(inpt["NUM_RES_BLOCKS"])
except:
    NUM_RES_BLOCKS = 0
if NUM_RES_BLOCKS > 0:
    NUM_RES_BLOCK_LAYERS = int(inpt["NUM_RES_BLOCK_LAYERS"])
    NUM_RES_BLOCK_UNITS = int(inpt["NUM_RES_BLOCK_UNITS"])
else:
    NUM_RES_BLOCK_LAYERS = 1
    NUM_RES_BLOCK_UNITS = 1

try:
    LOAD_MODEL = inpt["LOAD_MODEL"]
    if not os.path.isfile(LOAD_MODEL):
        LOAD_MODEL = None
except KeyError:
    LOAD_MODEL = None

if MERGED:
    hidden_units_t = [NEURONS_NUM] * LAYERS_T_NUM
    hidden_units_t_r = [NEURONS_NUM] * LAYERS_TR_NUM
    hidden_units_cs_a = [NEURONS_NUM] * LAYERS_TR_VAR_NUM
    hidden_units_cs_c = [NEURONS_NUM] * LAYERS_TR_VAR_NUM
    hidden_units_phie = [NEURONS_NUM] * LAYERS_T_VAR_NUM
    hidden_units_phis_c = [NEURONS_NUM] * LAYERS_T_VAR_NUM

else:
    hidden_units_t = None
    hidden_units_t_r = None
    hidden_units_cs_a = [NEURONS_NUM] * LAYERS_SPLIT_NUM
    hidden_units_cs_c = [NEURONS_NUM] * LAYERS_SPLIT_NUM
    hidden_units_phie = [NEURONS_NUM] * LAYERS_SPLIT_NUM
    hidden_units_phis_c = [NEURONS_NUM] * LAYERS_SPLIT_NUM


if not dataFolder is None and alpha[2] > 0:
    # data_phie = np.load(os.path.join(dataFolder, "data_phie.npz"))
    data_phie = np.load(os.path.join(dataFolder, "data_phie_multi.npz"))
    xTrain_phie = data_phie["x_train"].astype("float64")
    yTrain_phie = data_phie["y_train"].astype("float64")
    x_params_train_phie = data_phie["x_params_train"].astype("float64")
    # data_phis_c = np.load(os.path.join(dataFolder, "data_phis_c.npz"))
    data_phis_c = np.load(os.path.join(dataFolder, "data_phis_c_multi.npz"))
    xTrain_phis_c = data_phis_c["x_train"].astype("float64")
    yTrain_phis_c = data_phis_c["y_train"].astype("float64")
    x_params_train_phis_c = data_phis_c["x_params_train"].astype("float64")
    # data_cs_a = np.load(os.path.join(dataFolder, "data_cs_a.npz"))
    data_cs_a = np.load(os.path.join(dataFolder, "data_cs_a_multi.npz"))
    xTrain_cs_a = data_cs_a["x_train"].astype("float64")
    yTrain_cs_a = data_cs_a["y_train"].astype("float64")
    x_params_train_cs_a = data_cs_a["x_params_train"].astype("float64")
    # data_cs_c = np.load(os.path.join(dataFolder, "data_cs_c.npz"))
    data_cs_c = np.load(os.path.join(dataFolder, "data_cs_c_multi.npz"))
    xTrain_cs_c = data_cs_c["x_train"].astype("float64")
    yTrain_cs_c = data_cs_c["y_train"].astype("float64")
    x_params_train_cs_c = data_cs_c["x_params_train"].astype("float64")
else:
    nParams = 2
    print("INFO: LOADING DUMMY DATA")
    # Dummy data
    xTrain_phie = np.zeros((N_BATCH, 1)).astype("float64")
    x_params_train_phie = np.zeros((N_BATCH, nParams)).astype("float64")
    yTrain_phie = np.zeros((N_BATCH, 1)).astype("float64")
    xTrain_phis_c = np.zeros((N_BATCH, 1)).astype("float64")
    x_params_train_phis_c = np.zeros((N_BATCH, nParams)).astype("float64")
    yTrain_phis_c = np.zeros((N_BATCH, 1)).astype("float64")
    xTrain_cs_a = np.zeros((N_BATCH, 2)).astype("float64")
    x_params_train_cs_a = np.zeros((N_BATCH, nParams)).astype("float64")
    yTrain_cs_a = np.zeros((N_BATCH, 1)).astype("float64")
    xTrain_cs_c = np.zeros((N_BATCH, 2)).astype("float64")
    x_params_train_cs_c = np.zeros((N_BATCH, nParams)).astype("float64")
    yTrain_cs_c = np.zeros((N_BATCH, 1)).astype("float64")

factorSchedulerModel = np.log(
    LEARNING_RATE_MODEL_FINAL / LEARNING_RATE_MODEL
) / ((EPOCHS + 1e-16) / 2)


def schedulerModel(epoch, lr):
    if epoch < EPOCHS // 2:
        return lr
    else:
        return max(
            lr * tf.math.exp(factorSchedulerModel), LEARNING_RATE_MODEL_FINAL
        )


factorSchedulerWeights = np.log(
    LEARNING_RATE_WEIGHTS_FINAL / LEARNING_RATE_WEIGHTS
) / ((EPOCHS + 1e-16) / 2)


def schedulerWeights(epoch, lr):
    if epoch < EPOCHS // 2:
        return lr
    else:
        return max(
            lr * tf.math.exp(factorSchedulerWeights),
            LEARNING_RATE_WEIGHTS_FINAL,
        )


nn = pinn(
    params=params,
    hidden_units_t=hidden_units_t,
    hidden_units_t_r=hidden_units_t_r,
    hidden_units_phie=hidden_units_phie,
    hidden_units_phis_c=hidden_units_phis_c,
    hidden_units_cs_a=hidden_units_cs_a,
    hidden_units_cs_c=hidden_units_cs_c,
    n_hidden_res_blocks=NUM_RES_BLOCKS,
    n_res_block_layers=NUM_RES_BLOCK_LAYERS,
    n_res_block_units=NUM_RES_BLOCK_UNITS,
    alpha=alpha,
    batch_size_int=BATCH_SIZE_INT,
    batch_size_bound=BATCH_SIZE_BOUND,
    batch_size_reg=BATCH_SIZE_REG,
    batch_size_struct=BATCH_SIZE_STRUCT,
    max_batch_size_data=MAX_BATCH_SIZE_DATA,
    n_batch=N_BATCH,
    n_batch_lbfgs=N_BATCH_LBFGS,
    hard_IC_timescale=np.float64(HARD_IC_TIMESCALE),
    exponentialLimiter=EXP_LIMITER,
    collocationMode=COLLOCATION_MODE,
    gradualTime=GRADUAL_TIME,
    firstTime=np.float64(HARD_IC_TIMESCALE * RATIO_FIRST_TIME),
    tmin_int_bound=np.float64(HARD_IC_TIMESCALE * RATIO_T_MIN),
    nEpochs=EPOCHS,
    nEpochs_lbfgs=EPOCHS_LBFGS,
    nEpochs_start_lbfgs=EPOCHS_START_LBFGS,
    initialLossThreshold=np.float64(LOSS_THRESHOLD),
    dynamicAttentionWeights=DYNAMIC_ATTENTION_WEIGHTS,
    useLossThreshold=USE_LOSS_THRESHOLD,
    activation=ACTIVATION,
    lbfgs=LBFGS,
    sgd=SGD,
    linearizeJ=LINEARIZE_J,
    params_min=[
        params["deg_i0_a_min"],
        params["deg_ds_c_min"],
    ],
    params_max=[
        params["deg_i0_a_max"],
        params["deg_ds_c_max"],
    ],
    xDataList=[
        xTrain_phie,
        xTrain_phis_c,
        xTrain_cs_a,
        xTrain_cs_c,
    ],
    x_params_dataList=[
        x_params_train_phie,
        x_params_train_phis_c,
        x_params_train_cs_a,
        x_params_train_cs_c,
    ],
    yDataList=[
        yTrain_phie,
        yTrain_phis_c,
        yTrain_cs_a,
        yTrain_cs_c,
    ],
    logLossFolder="Log_" + str(ID),
    modelFolder="Model_" + str(ID),
)

# plot_model(nn.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
if not LOAD_MODEL is None:
    print("INFO: Loading model %s" % LOAD_MODEL)
    nn.model.load_weights(LOAD_MODEL)
# nn.model.load_weights("saveModel_short/last.h5")

# if nn.activeInt:
#    sa_int_weights = np.load(
#        "Model.save/sa_int_weights.npy", allow_pickle=True
#    )
#    t_int_col = np.load("Model.save/t_int_col.npy")
#    r_a_int_col = np.load("Model.save/r_a_int_col.npy")
#    r_c_int_col = np.load("Model.save/r_c_int_col.npy")
#    r_maxa_int_col = np.load("Model.save/r_maxa_int_col.npy")
#    r_maxc_int_col = np.load("Model.save/r_maxc_int_col.npy")
# if nn.activeBound:
#    sa_bound_weights = np.load(
#        "Model.save/sa_bound_weights.npy", allow_pickle=True
#    )
#    t_bound_col = np.load("Model.save/t_bound_col.npy")
#    r_min_bound_col = np.load("Model.save/r_min_bound_col.npy")
#    r_maxa_bound_col = np.load("Model.save/r_maxa_bound_col.npy")
#    r_maxc_bound_col = np.load("Model.save/r_maxc_bound_col.npy")
# if nn.activeData:
#    sa_data_weights = np.load(
#        "Model.save/sa_data_weights.npy", allow_pickle=True
#    )
# if nn.activeReg:
#    sa_reg_weights = np.load(
#        "Model.save/sa_reg_weights.npy", allow_pickle=True
#    )
#    t_reg_col = np.load("Model.save/t_reg_col.npy")
#
# nn.loadCol(
#    t_int_col,
#    r_a_int_col,
#    r_c_int_col,
#    r_maxa_int_col,
#    r_maxc_int_col,
#    sa_int_weights,
#    t_bound_col,
#    r_min_bound_col,
#    r_maxa_bound_col,
#    r_maxc_bound_col,
#    sa_bound_weights,
# )

nn.train(
    learningRateModel=LEARNING_RATE_MODEL,
    learningRateModelFinal=LEARNING_RATE_MODEL_FINAL,
    lrSchedulerModel=schedulerModel,
    learningRateWeights=LEARNING_RATE_WEIGHTS,
    learningRateWeightsFinal=LEARNING_RATE_WEIGHTS_FINAL,
    lrSchedulerWeights=schedulerWeights,
    learningRateLBFGS=LEARNING_RATE_LBFGS,
    inner_epochs=INNER_EPOCHS,
    start_weight_training_epoch=START_WEIGHT_TRAINING_EPOCH,
)

shutil.copytree(nn.modelFolder, "ModelFin_" + str(ID))
shutil.copytree(nn.logLossFolder, "LogFin_" + str(ID))
