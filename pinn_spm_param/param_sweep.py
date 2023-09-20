import os
import random

import numpy as np


def rewriteInput(param, paramName, inputPathRef, inputPath):
    # Read inputFile
    f = open(inputPathRef, "r")
    lines = f.readlines()
    f.close()
    # Write inputFile
    f = open(inputPath, "w")
    for line in lines:
        foundFlag = False
        for ivar, varname in enumerate(paramName):
            if line.startswith(varname + " :"):
                f.write(varname + " : " + str(param[ivar]) + "\n")
                foundFlag = True
        if not foundFlag:
            f.write(line)
    f.close()
    return


EPOCHS_list = [100, 200, 400]
DYNAMIC_ATTENTION_WEIGHTS_list = ["True", "False"]
USE_LOSS_THRESHOLD_list = ["True", "False"]
LEARNING_RATE_MODEL_list = [5e-3, 5e-4]
# LEARNING_RATE_MODEL_FINAL = LEARNING_RATE_MODEL /5
BATCH_SIZE_INT_list = [64, 128, 256]
N_BATCH_list = [10]
HARD_IC_TIMESCALE_list = [0.01, 0.1, 1]
NEURONS_NUM_list = [2, 4, 8, 16]
LAYERS_TRUNK_list = [2, 4, 8]
LAYERS_VAR_list = [1, 2, 4]
RATIO_T_MIN_list = [0.6, 1.1, 2.1, 4.1, 8.1]
# RATIO_FIRST_TIME = RATIO_T_MIN * 2

TotalList = []
for a in DYNAMIC_ATTENTION_WEIGHTS_list:
    for b in USE_LOSS_THRESHOLD_list:
        for c in LEARNING_RATE_MODEL_list:
            for d in BATCH_SIZE_INT_list:
                for f in N_BATCH_list:
                    for g in HARD_IC_TIMESCALE_list:
                        for h in NEURONS_NUM_list:
                            for i in LAYERS_TRUNK_list:
                                for j in LAYERS_VAR_list:
                                    for k in RATIO_T_MIN_list:
                                        for l in EPOCHS_list:
                                            TotalList.append(
                                                [
                                                    a,
                                                    b,
                                                    c,
                                                    d,
                                                    f,
                                                    g,
                                                    h,
                                                    i,
                                                    j,
                                                    k,
                                                    l,
                                                ]
                                            )

nSim = 10000
Sweep = random.sample(TotalList, nSim)
np.save("params.npy", Sweep)


params_name = [
    "ID",
    "DYNAMIC_ATTENTION_WEIGHTS",
    "USE_LOSS_THRESHOLD",
    "LEARNING_RATE_MODEL",
    "LEARNING_RATE_MODEL_FINAL",
    "BATCH_SIZE_INT",
    "BATCH_SIZE_BOUND",
    "N_BATCH",
    "HARD_IC_TIMESCALE",
    "NEURONS_NUM",
    "LAYERS_T_NUM",
    "LAYERS_TR_NUM",
    "LAYERS_T_VAR_NUM",
    "LAYERS_TR_VAR_NUM",
    "RATIO_FIRST_TIME",
    "RATIO_T_MIN",
    "EPOCHS",
]

input_dir = "input_sweep"
os.makedirs(input_dir, exist_ok=True)

for isim, sweep in enumerate(Sweep):
    params_val = [
        isim,
        sweep[0],
        sweep[1],
        sweep[2],
        sweep[2] / 5,
        sweep[3],
        int(sweep[3] / 2),
        sweep[4],
        sweep[5],
        sweep[6],
        sweep[7],
        sweep[7],
        sweep[8],
        sweep[8],
        sweep[9] * 2,
        sweep[9],
        sweep[10],
    ]
    file_to_write = os.path.join(input_dir, "input" + str(isim))
    rewriteInput(params_val, params_name, "input", file_to_write)
