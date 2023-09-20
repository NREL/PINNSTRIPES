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
from init_pinn import (
    initialize_nn,
    initialize_params,
    initialize_params_from_inpt,
)
from myNN import *
from myparser import parseInputFile
from myProgressBar import printProgressBar


def do_training_only(input_params, nn):
    LEARNING_RATE_LBFGS = input_params["LEARNING_RATE_LBFGS"]
    LEARNING_RATE_MODEL = input_params["LEARNING_RATE_MODEL"]
    LEARNING_RATE_MODEL_FINAL = input_params["LEARNING_RATE_MODEL_FINAL"]
    LEARNING_RATE_WEIGHTS = input_params["LEARNING_RATE_WEIGHTS"]
    LEARNING_RATE_WEIGHTS_FINAL = input_params["LEARNING_RATE_WEIGHTS_FINAL"]
    GRADIENT_THRESHOLD = input_params["GRADIENT_THRESHOLD"]
    INNER_EPOCHS = input_params["INNER_EPOCHS"]
    EPOCHS = input_params["EPOCHS"]
    START_WEIGHT_TRAINING_EPOCH = input_params["START_WEIGHT_TRAINING_EPOCH"]
    ID = input_params["ID"]

    factorSchedulerModel = np.log(
        LEARNING_RATE_MODEL_FINAL / (LEARNING_RATE_MODEL + 1e-16)
    ) / ((EPOCHS + 1e-16) / 2)

    def schedulerModel(epoch, lr):
        if epoch < EPOCHS // 2:
            return lr
        else:
            return max(
                lr * tf.math.exp(factorSchedulerModel),
                LEARNING_RATE_MODEL_FINAL,
            )

    factorSchedulerWeights = np.log(
        LEARNING_RATE_WEIGHTS_FINAL / (LEARNING_RATE_WEIGHTS + 1e-16)
    ) / ((EPOCHS + 1e-16) / 2)

    def schedulerWeights(epoch, lr):
        if epoch < EPOCHS // 2:
            return lr
        else:
            return max(
                lr * tf.math.exp(factorSchedulerWeights),
                LEARNING_RATE_WEIGHTS_FINAL,
            )

    time_start = time.time()
    unweighted_loss = nn.train(
        learningRateModel=LEARNING_RATE_MODEL,
        learningRateModelFinal=LEARNING_RATE_MODEL_FINAL,
        lrSchedulerModel=schedulerModel,
        learningRateWeights=LEARNING_RATE_WEIGHTS,
        learningRateWeightsFinal=LEARNING_RATE_WEIGHTS_FINAL,
        lrSchedulerWeights=schedulerWeights,
        learningRateLBFGS=LEARNING_RATE_LBFGS,
        inner_epochs=INNER_EPOCHS,
        start_weight_training_epoch=START_WEIGHT_TRAINING_EPOCH,
        gradient_threshold=GRADIENT_THRESHOLD,
    )
    time_end = time.time()

    return time_end - time_start, unweighted_loss


def do_training(input_params, nn):
    ID = input_params["ID"]
    elapsedTime, unweighted_loss = do_training_only(input_params, nn)
    shutil.copytree(nn.modelFolder, "ModelFin_" + str(ID))
    shutil.copytree(nn.logLossFolder, "LogFin_" + str(ID))
    return elapsedTime, unweighted_loss


def main():
    # Read command line arguments
    args = argument.initArg()
    input_params = initialize_params(args)
    nn = initialize_nn(args=args, input_params=input_params)
    elapsed, unweighted_loss = do_training(input_params=input_params, nn=nn)
    print(f"Total time {elapsed:.2f}s")
    print(f"Unweighted loss {unweighted_loss}")


if __name__ == "__main__":
    main()
