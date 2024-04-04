import os
import sys
import time

import numpy as np

sys.path.append("util")
import shutil

import argument

# NN Stuff
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from myNN import *
from prettyPlot.parser import parse_input_file


def absolute_path_check(path):
    if path is None:
        absolute = True
    elif os.path.isabs(path):
        absolute = True
    elif not os.path.isabs(path):
        absolute = False
    if not absolute:
        sys.exit(f"ERROR: {path} is not absolute")


def safe_load(nn, weight_path):
    loaded = False
    ntry = 0
    while not loaded:
        try:
            nn.model.load_weights(weight_path)
            loaded = True
        except:
            ntry += 1
        if ntry > 1000:
            sys.exit(f"ERROR: could not load {weight_path}")
    return nn


def initialize_params_from_inpt(inpt):
    try:
        seed = int(inpt["seed"])
    except KeyError:
        seed = -1
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
    try:
        GRADIENT_THRESHOLD = float(inpt["GRADIENT_THRESHOLD"])
    except KeyError:
        GRADIENT_THRESHOLD = None
    HARD_IC_TIMESCALE = float(inpt["HARD_IC_TIMESCALE"])
    RATIO_FIRST_TIME = float(inpt["RATIO_FIRST_TIME"])
    RATIO_T_MIN = float(inpt["RATIO_T_MIN"])
    EXP_LIMITER = float(inpt["EXP_LIMITER"])
    COLLOCATION_MODE = inpt["COLLOCATION_MODE"]
    GRADUAL_TIME_SGD = inpt["GRADUAL_TIME_SGD"] == "True"
    GRADUAL_TIME_LBFGS = inpt["GRADUAL_TIME_LBFGS"] == "True"
    FIRST_TIME_LBFGS = None
    N_GRADUAL_STEPS_LBFGS = None
    GRADUAL_TIME_MODE_LBFGS = None
    if GRADUAL_TIME_LBFGS:
        N_GRADUAL_STEPS_LBFGS = int(inpt["N_GRADUAL_STEPS_LBFGS"])
        try:
            GRADUAL_TIME_MODE_LBFGS = inpt["GRADUAL_TIME_MODE_LBFGS"]
        except KeyError:
            GRADUAL_TIME_MODE_LBFGS = "linear"

    DYNAMIC_ATTENTION_WEIGHTS = inpt["DYNAMIC_ATTENTION_WEIGHTS"] == "True"
    ANNEALING_WEIGHTS = inpt["ANNEALING_WEIGHTS"] == "True"
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
    try:
        LOCAL_utilFolder = inpt["LOCAL_utilFolder"]
    except KeyError:
        LOCAL_utilFolder = os.path.join(os.getcwd(), "util")
    try:
        HNN_utilFolder = inpt["HNN_utilFolder"]
    except KeyError:
        HNN_utilFolder = None
    try:
        HNN_modelFolder = inpt["HNN_modelFolder"]
    except KeyError:
        HNN_modelFolder = None
    try:
        HNN_params = [
            np.float64(entry) for entry in inpt["HNN_params"].split()
        ]
    except KeyError:
        HNN_params = None
    if (
        HNN_utilFolder is None
        or HNN_modelFolder is None
        or not os.path.isdir(HNN_utilFolder)
        or not os.path.isdir(HNN_modelFolder)
    ):
        HNN_utilFolder = None
        HNN_modelFolder = None
        HNN_params = None
    try:
        HNNTIME_utilFolder = inpt["HNNTIME_utilFolder"]
    except KeyError:
        HNNTIME_utilFolder = None
    try:
        HNNTIME_modelFolder = inpt["HNNTIME_modelFolder"]
    except KeyError:
        HNNTIME_modelFolder = None
    try:
        HNNTIME_val = np.float64(inpt["HNNTIME_val"])
    except (KeyError, ValueError):
        HNNTIME_val = None
    if (
        HNNTIME_utilFolder is None
        or HNNTIME_modelFolder is None
        or HNNTIME_val is None
        or not os.path.isdir(HNNTIME_utilFolder)
        or not os.path.isdir(HNNTIME_modelFolder)
    ):
        HNNTIME_utilFolder = None
        HNNTIME_modelFolder = None
        HNNTIME_val = None

    if (HNN_utilFolder is not None) or (HNNTIME_utilFolder is not None):
        if not os.path.isdir(LOCAL_utilFolder):
            print(f"ERROR: {LOCAL_utilFolder} is not a directory")
            sys.exit()
    absolute_path_check(LOCAL_utilFolder)
    absolute_path_check(HNN_utilFolder)
    absolute_path_check(HNN_modelFolder)
    absolute_path_check(HNNTIME_utilFolder)
    absolute_path_check(HNNTIME_modelFolder)

    ACTIVATION = inpt["ACTIVATION"]
    LBFGS = inpt["LBFGS"] == "True"
    SGD = inpt["SGD"] == "True"
    MERGED = inpt["MERGED"] == "True"
    LINEARIZE_J = inpt["LINEARIZE_J"] == "True"

    # WEIGHTING
    try:
        weights = {}
        weights["phie_int"] = np.float64(inpt["w_phie_int"])
        weights["phis_c_int"] = np.float64(inpt["w_phis_c_int"])
        weights["cs_a_int"] = np.float64(inpt["w_cs_a_int"])
        weights["cs_c_int"] = np.float64(inpt["w_cs_c_int"])

        weights["cs_a_rmin_bound"] = np.float64(inpt["w_cs_a_rmin_bound"])
        weights["cs_a_rmax_bound"] = np.float64(inpt["w_cs_a_rmax_bound"])
        weights["cs_c_rmin_bound"] = np.float64(inpt["w_cs_c_rmin_bound"])
        weights["cs_c_rmax_bound"] = np.float64(inpt["w_cs_c_rmax_bound"])

        weights["phie_dat"] = np.float64(inpt["w_phie_dat"])
        weights["phis_c_dat"] = np.float64(inpt["w_phis_c_dat"])
        weights["cs_a_dat"] = np.float64(inpt["w_cs_a_dat"])
        weights["cs_c_dat"] = np.float64(inpt["w_cs_c_dat"])
    except KeyError:
        weights = None

    # Surrogate NN
    BATCH_SIZE_INT = int(inpt["BATCH_SIZE_INT"])
    BATCH_SIZE_BOUND = int(inpt["BATCH_SIZE_BOUND"])
    MAX_BATCH_SIZE_DATA = int(inpt["MAX_BATCH_SIZE_DATA"])
    BATCH_SIZE_REG = int(inpt["BATCH_SIZE_REG"])
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
        NUM_GRAD_PATH_LAYERS = int(inpt["NUM_GRAD_PATH_LAYERS"])
    except:
        NUM_GRAD_PATH_LAYERS = None
    if NUM_GRAD_PATH_LAYERS is not None:
        NUM_GRAD_PATH_UNITS = int(inpt["NUM_GRAD_PATH_UNITS"])
    else:
        NUM_GRAD_PATH_UNITS = None

    try:
        LOAD_MODEL = inpt["LOAD_MODEL"]
        if not os.path.isfile(LOAD_MODEL):
            LOAD_MODEL = None
    except KeyError:
        LOAD_MODEL = None

    return {
        "MERGED": MERGED,
        "NEURONS_NUM": NEURONS_NUM,
        "LAYERS_T_NUM": LAYERS_T_NUM,
        "LAYERS_TR_NUM": LAYERS_TR_NUM,
        "LAYERS_T_VAR_NUM": LAYERS_T_VAR_NUM,
        "LAYERS_TR_VAR_NUM": LAYERS_TR_VAR_NUM,
        "LAYERS_SPLIT_NUM": LAYERS_SPLIT_NUM,
        "seed": seed,
        "ID": ID,
        "EPOCHS": EPOCHS,
        "EPOCHS_LBFGS": EPOCHS_LBFGS,
        "EPOCHS_START_LBFGS": EPOCHS_START_LBFGS,
        "alpha": alpha,
        "LEARNING_RATE_WEIGHTS": LEARNING_RATE_WEIGHTS,
        "LEARNING_RATE_WEIGHTS_FINAL": LEARNING_RATE_WEIGHTS_FINAL,
        "LEARNING_RATE_MODEL": LEARNING_RATE_MODEL,
        "LEARNING_RATE_MODEL_FINAL": LEARNING_RATE_MODEL_FINAL,
        "LEARNING_RATE_LBFGS": LEARNING_RATE_LBFGS,
        "GRADIENT_THRESHOLD": GRADIENT_THRESHOLD,
        "HARD_IC_TIMESCALE": HARD_IC_TIMESCALE,
        "RATIO_FIRST_TIME": RATIO_FIRST_TIME,
        "RATIO_T_MIN": RATIO_T_MIN,
        "EXP_LIMITER": EXP_LIMITER,
        "COLLOCATION_MODE": COLLOCATION_MODE,
        "GRADUAL_TIME_SGD": GRADUAL_TIME_SGD,
        "GRADUAL_TIME_LBFGS": GRADUAL_TIME_LBFGS,
        "FIRST_TIME_LBFGS": FIRST_TIME_LBFGS,
        "N_GRADUAL_STEPS_LBFGS": N_GRADUAL_STEPS_LBFGS,
        "GRADUAL_TIME_MODE_LBFGS": GRADUAL_TIME_MODE_LBFGS,
        "DYNAMIC_ATTENTION_WEIGHTS": DYNAMIC_ATTENTION_WEIGHTS,
        "ANNEALING_WEIGHTS": ANNEALING_WEIGHTS,
        "USE_LOSS_THRESHOLD": USE_LOSS_THRESHOLD,
        "LOSS_THRESHOLD": LOSS_THRESHOLD,
        "INNER_EPOCHS": INNER_EPOCHS,
        "START_WEIGHT_TRAINING_EPOCH": START_WEIGHT_TRAINING_EPOCH,
        "HNN_utilFolder": HNN_utilFolder,
        "HNN_modelFolder": HNN_modelFolder,
        "HNN_params": HNN_params,
        "HNNTIME_utilFolder": HNNTIME_utilFolder,
        "HNNTIME_modelFolder": HNNTIME_modelFolder,
        "HNNTIME_val": HNNTIME_val,
        "ACTIVATION": ACTIVATION,
        "LBFGS": LBFGS,
        "SGD": SGD,
        "LINEARIZE_J": LINEARIZE_J,
        "weights": weights,
        "BATCH_SIZE_INT": BATCH_SIZE_INT,
        "BATCH_SIZE_BOUND": BATCH_SIZE_BOUND,
        "MAX_BATCH_SIZE_DATA": MAX_BATCH_SIZE_DATA,
        "BATCH_SIZE_REG": BATCH_SIZE_REG,
        "N_BATCH": N_BATCH,
        "N_BATCH_LBFGS": N_BATCH_LBFGS,
        "NUM_RES_BLOCKS": NUM_RES_BLOCKS,
        "NUM_RES_BLOCK_LAYERS": NUM_RES_BLOCK_LAYERS,
        "NUM_RES_BLOCK_UNITS": NUM_RES_BLOCK_UNITS,
        "NUM_GRAD_PATH_LAYERS": NUM_GRAD_PATH_LAYERS,
        "NUM_GRAD_PATH_UNITS": NUM_GRAD_PATH_UNITS,
        "LOAD_MODEL": LOAD_MODEL,
        "LOCAL_utilFolder": LOCAL_utilFolder,
    }


def initialize_params(args):
    inpt = parse_input_file(args.input_file)
    return initialize_params_from_inpt(inpt)


def initialize_nn(args, input_params):
    seed = input_params["seed"]
    NEURONS_NUM = input_params["NEURONS_NUM"]
    LAYERS_T_NUM = input_params["LAYERS_T_NUM"]
    LAYERS_TR_NUM = input_params["LAYERS_TR_NUM"]
    LAYERS_T_VAR_NUM = input_params["LAYERS_T_VAR_NUM"]
    LAYERS_TR_VAR_NUM = input_params["LAYERS_TR_VAR_NUM"]
    LAYERS_SPLIT_NUM = input_params["LAYERS_SPLIT_NUM"]
    alpha = input_params["alpha"]
    N_BATCH = input_params["N_BATCH"]
    LEARNING_RATE_MODEL = input_params["LEARNING_RATE_MODEL"]
    LEARNING_RATE_MODEL_FINAL = input_params["LEARNING_RATE_MODEL_FINAL"]
    LEARNING_RATE_WEIGHTS = input_params["LEARNING_RATE_WEIGHTS"]
    LEARNING_RATE_WEIGHTS_FINAL = input_params["LEARNING_RATE_WEIGHTS_FINAL"]
    GRADIENT_THRESHOLD = input_params["GRADIENT_THRESHOLD"]
    EPOCHS = input_params["EPOCHS"]
    NUM_RES_BLOCKS = input_params["NUM_RES_BLOCKS"]
    NUM_RES_BLOCK_LAYERS = input_params["NUM_RES_BLOCK_LAYERS"]
    NUM_RES_BLOCK_UNITS = input_params["NUM_RES_BLOCK_UNITS"]
    NUM_GRAD_PATH_LAYERS = input_params["NUM_GRAD_PATH_LAYERS"]
    NUM_GRAD_PATH_UNITS = input_params["NUM_GRAD_PATH_UNITS"]
    BATCH_SIZE_INT = input_params["BATCH_SIZE_INT"]
    BATCH_SIZE_BOUND = input_params["BATCH_SIZE_BOUND"]
    BATCH_SIZE_REG = input_params["BATCH_SIZE_REG"]
    MAX_BATCH_SIZE_DATA = input_params["MAX_BATCH_SIZE_DATA"]
    N_BATCH_LBFGS = input_params["N_BATCH_LBFGS"]
    HARD_IC_TIMESCALE = input_params["HARD_IC_TIMESCALE"]
    EXP_LIMITER = input_params["EXP_LIMITER"]
    COLLOCATION_MODE = input_params["COLLOCATION_MODE"]
    GRADUAL_TIME_SGD = input_params["GRADUAL_TIME_SGD"]
    GRADUAL_TIME_LBFGS = input_params["GRADUAL_TIME_LBFGS"]
    GRADUAL_TIME_MODE_LBFGS = input_params["GRADUAL_TIME_MODE_LBFGS"]
    RATIO_FIRST_TIME = input_params["RATIO_FIRST_TIME"]
    N_GRADUAL_STEPS_LBFGS = input_params["N_GRADUAL_STEPS_LBFGS"]
    RATIO_T_MIN = input_params["RATIO_T_MIN"]
    EPOCHS_LBFGS = input_params["EPOCHS_LBFGS"]
    EPOCHS_START_LBFGS = input_params["EPOCHS_START_LBFGS"]
    LOSS_THRESHOLD = input_params["LOSS_THRESHOLD"]
    DYNAMIC_ATTENTION_WEIGHTS = input_params["DYNAMIC_ATTENTION_WEIGHTS"]
    ANNEALING_WEIGHTS = input_params["ANNEALING_WEIGHTS"]
    USE_LOSS_THRESHOLD = input_params["USE_LOSS_THRESHOLD"]
    ACTIVATION = input_params["ACTIVATION"]
    LBFGS = input_params["LBFGS"]
    SGD = input_params["SGD"]
    LINEARIZE_J = input_params["LINEARIZE_J"]
    LOAD_MODEL = input_params["LOAD_MODEL"]
    MERGED = input_params["MERGED"]
    ID = input_params["ID"]
    LOCAL_utilFolder = input_params["LOCAL_utilFolder"]
    HNN_modelFolder = input_params["HNN_modelFolder"]
    HNN_utilFolder = input_params["HNN_utilFolder"]
    HNN_params = input_params["HNN_params"]
    HNNTIME_modelFolder = input_params["HNNTIME_modelFolder"]
    HNNTIME_utilFolder = input_params["HNNTIME_utilFolder"]
    HNNTIME_val = input_params["HNNTIME_val"]
    weights = input_params["weights"]

    if args.simpleModel:
        from spm_simpler import makeParams
    else:
        from spm import makeParams

    params = makeParams()
    dataFolder = args.dataFolder

    if seed >= 0:
        tf.random.set_seed(seed)
        np.random.seed(seed)

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

    if dataFolder is not None and os.path.isdir(dataFolder) and alpha[2] > 0:
        try:
            data_phie = np.load(
                os.path.join(dataFolder, "data_phie_multi.npz")
            )
            use_multi = True
            print("INFO: LOADING MULTI DATASETS")
        except:
            data_phie = np.load(os.path.join(dataFolder, "data_phie.npz"))
            use_multi = False
            print("INFO: LOADING SINGLE DATASETS")
        xTrain_phie = data_phie["x_train"].astype("float64")
        yTrain_phie = data_phie["y_train"].astype("float64")
        x_params_train_phie = data_phie["x_params_train"].astype("float64")

        if use_multi:
            data_phis_c = np.load(
                os.path.join(dataFolder, "data_phis_c_multi.npz")
            )
        else:
            data_phis_c = np.load(os.path.join(dataFolder, "data_phis_c.npz"))
        xTrain_phis_c = data_phis_c["x_train"].astype("float64")
        yTrain_phis_c = data_phis_c["y_train"].astype("float64")
        x_params_train_phis_c = data_phis_c["x_params_train"].astype("float64")
        if use_multi:
            data_cs_a = np.load(
                os.path.join(dataFolder, "data_cs_a_multi.npz")
            )
        else:
            data_cs_a = np.load(os.path.join(dataFolder, "data_cs_a.npz"))
        xTrain_cs_a = data_cs_a["x_train"].astype("float64")
        yTrain_cs_a = data_cs_a["y_train"].astype("float64")
        x_params_train_cs_a = data_cs_a["x_params_train"].astype("float64")
        if use_multi:
            data_cs_c = np.load(
                os.path.join(dataFolder, "data_cs_c_multi.npz")
            )
        else:
            data_cs_c = np.load(os.path.join(dataFolder, "data_cs_c.npz"))
        xTrain_cs_c = data_cs_c["x_train"].astype("float64")
        yTrain_cs_c = data_cs_c["y_train"].astype("float64")
        x_params_train_cs_c = data_cs_c["x_params_train"].astype("float64")
    else:
        nParams = 2
        print("INFO: LOADING DUMMY DATA")
        # Dummy data
        xTrain_phie = np.zeros((N_BATCH, 1)).astype("float64")
        yTrain_phie = np.zeros((N_BATCH, 1)).astype("float64")
        x_params_train_phie = np.zeros((N_BATCH, nParams)).astype("float64")
        xTrain_phis_c = np.zeros((N_BATCH, 1)).astype("float64")
        yTrain_phis_c = np.zeros((N_BATCH, 1)).astype("float64")
        x_params_train_phis_c = np.zeros((N_BATCH, nParams)).astype("float64")
        xTrain_cs_a = np.zeros((N_BATCH, 2)).astype("float64")
        yTrain_cs_a = np.zeros((N_BATCH, 1)).astype("float64")
        x_params_train_cs_a = np.zeros((N_BATCH, nParams)).astype("float64")
        xTrain_cs_c = np.zeros((N_BATCH, 2)).astype("float64")
        yTrain_cs_c = np.zeros((N_BATCH, 1)).astype("float64")
        x_params_train_cs_c = np.zeros((N_BATCH, nParams)).astype("float64")

    nn = myNN(
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
        n_grad_path_layers=NUM_GRAD_PATH_LAYERS,
        n_grad_path_units=NUM_GRAD_PATH_UNITS,
        alpha=alpha,
        batch_size_int=BATCH_SIZE_INT,
        batch_size_bound=BATCH_SIZE_BOUND,
        batch_size_reg=BATCH_SIZE_REG,
        max_batch_size_data=MAX_BATCH_SIZE_DATA,
        n_batch=N_BATCH,
        n_batch_lbfgs=N_BATCH_LBFGS,
        hard_IC_timescale=np.float64(HARD_IC_TIMESCALE),
        exponentialLimiter=EXP_LIMITER,
        collocationMode=COLLOCATION_MODE,
        gradualTime_sgd=GRADUAL_TIME_SGD,
        gradualTime_lbfgs=GRADUAL_TIME_LBFGS,
        gradualTimeMode_lbfgs=GRADUAL_TIME_MODE_LBFGS,
        firstTime=np.float64(HARD_IC_TIMESCALE * RATIO_FIRST_TIME),
        n_gradual_steps_lbfgs=N_GRADUAL_STEPS_LBFGS,
        tmin_int_bound=np.float64(HARD_IC_TIMESCALE * RATIO_T_MIN),
        nEpochs=EPOCHS,
        nEpochs_lbfgs=EPOCHS_LBFGS,
        nEpochs_start_lbfgs=EPOCHS_START_LBFGS,
        initialLossThreshold=np.float64(LOSS_THRESHOLD),
        dynamicAttentionWeights=DYNAMIC_ATTENTION_WEIGHTS,
        annealingWeights=ANNEALING_WEIGHTS,
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
        local_utilFolder=LOCAL_utilFolder,
        hnn_utilFolder=HNN_utilFolder,
        hnn_modelFolder=HNN_modelFolder,
        hnn_params=HNN_params,
        hnntime_utilFolder=HNNTIME_utilFolder,
        hnntime_modelFolder=HNNTIME_modelFolder,
        hnntime_val=HNNTIME_val,
        weights=weights,
        verbose=True,
    )

    if not args.optimized:
        import keras
        from keras.utils import plot_model

        try:
            plot_model(
                nn.model,
                to_file="model_plot.png",
                show_shapes=True,
                show_layer_names=True,
                show_layer_activations=False,
            )
        except:
            print("WARNING: Could not plot model")
    if not LOAD_MODEL is None:
        print("INFO: Loading model %s" % LOAD_MODEL)
        nn = safe_load(nn, LOAD_MODEL)

    return nn


def initialize_nn_from_params_config(params, configDict):
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
    try:
        n_grad_path_layers = configDict["n_grad_path_layers"]
    except:
        n_grad_path_layers = None
    if n_grad_path_layers is not None and n_grad_path_layers > 0:
        n_grad_path_units = configDict["n_grad_path_units"]
    else:
        n_grad_path_units = None
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
    try:
        local_utilFolder = configDict["local_utilFolder"]
    except KeyError:
        local_utilFolder = None
    try:
        hnn_utilFolder = configDict["hnn_utilFolder"]
    except KeyError:
        hnn_utilFolder = None
    try:
        hnn_modelFolder = configDict["hnn_modelFolder"]
    except KeyError:
        hnn_modelFolder = None
    try:
        hnn_params = configDict["hnn_params"]
    except KeyError:
        hnn_params = None
    try:
        hnntime_utilFolder = configDict["hnntime_utilFolder"]
    except KeyError:
        hnntime_utilFolder = None
    try:
        hnntime_modelFolder = configDict["hnntime_modelFolder"]
    except KeyError:
        hnntime_modelFolder = None
    try:
        hnntime_val = configDict["hnntime_val"]
    except KeyError:
        hnntime_val = None

    nn = myNN(
        params=params,
        hidden_units_t=hidden_units_t,
        hidden_units_t_r=hidden_units_t_r,
        hidden_units_phie=hidden_units_phie,
        hidden_units_phis_c=hidden_units_phis_c,
        hidden_units_cs_a=hidden_units_cs_a,
        hidden_units_cs_c=hidden_units_cs_c,
        n_hidden_res_blocks=n_hidden_res_blocks,
        n_res_block_layers=n_res_block_layers,
        n_res_block_units=n_res_block_units,
        n_grad_path_layers=n_grad_path_layers,
        n_grad_path_units=n_grad_path_units,
        hard_IC_timescale=np.float64(HARD_IC_TIMESCALE),
        exponentialLimiter=EXP_LIMITER,
        activation=ACTIVATION,
        linearizeJ=LINEARIZE_J,
        params_min=params_min,
        params_max=params_max,
        local_utilFolder=local_utilFolder,
        hnn_utilFolder=hnn_utilFolder,
        hnn_modelFolder=hnn_modelFolder,
        hnn_params=hnn_params,
        hnntime_utilFolder=hnntime_utilFolder,
        hnntime_modelFolder=hnntime_modelFolder,
        hnntime_val=hnntime_val,
        verbose=True,
    )

    return nn
