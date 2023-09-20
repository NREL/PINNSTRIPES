import sys
import time

import sherpa
from main import *

sys.path.append("postProcess")
from computeError import *


def main_optim_arch(trial, args, input_params):
    # Read command line arguments
    input_params["ID"] = trial.id
    input_params["NEURONS_NUM"] = trial.parameters["NEURONS_NUM"]
    input_params["LAYERS_T_NUM"] = trial.parameters["LAYERS_T_NUM"]
    input_params["LAYERS_TR_NUM"] = trial.parameters["LAYERS_TR_NUM"]
    input_params["LAYERS_T_VAR_NUM"] = trial.parameters["LAYERS_T_VAR_NUM"]
    input_params["LAYERS_TR_VAR_NUM"] = trial.parameters["LAYERS_TR_VAR_NUM"]
    input_params["NUM_RES_BLOCKS"] = trial.parameters["NUM_RES_BLOCKS"]
    input_params["NUM_RES_BLOCK_LAYERS"] = trial.parameters[
        "NUM_RES_BLOCK_LAYERS"
    ]
    input_params["NUM_RES_BLOCK_UNITS"] = trial.parameters[
        "NUM_RES_BLOCK_UNITS"
    ]
    input_params["ACTIVATION"] = trial.parameters["ACTIVATION"]
    nn = initialize_nn(args=args, input_params=input_params)
    elapsed, unweighted_loss = do_training(input_params=input_params, nn=nn)
    return elapsed, unweighted_loss


def main_optim_lossWeights(trial, args, input_params):
    # Read command line arguments
    input_params["ID"] = trial.id
    weights = {}
    weights["phie_int"] = trial.parameters["w_phi_int"]
    weights["phis_c_int"] = trial.parameters["w_phi_int"]
    weights["cs_a_int"] = trial.parameters["w_cs_int"]
    weights["cs_c_int"] = trial.parameters["w_cs_int"]

    weights["cs_a_rmin_bound"] = trial.parameters["w_cs_rmin_bound"]
    weights["cs_a_rmax_bound"] = trial.parameters["w_cs_rmax_bound"]
    weights["cs_c_rmin_bound"] = trial.parameters["w_cs_rmin_bound"]
    weights["cs_c_rmax_bound"] = trial.parameters["w_cs_rmax_bound"]

    # weights["phie_dat"] = trial.parameters["w_phie_dat"]
    # weights["phis_c_dat"] = trial.parameters["w_phis_c_dat"]
    # weights["cs_a_dat"] = trial.parameters["w_cs_a_dat"]
    # weights["cs_c_dat"] = trial.parameters["w_cs_c_dat"]
    weights["phie_dat"] = np.float64(1.0)
    weights["phis_c_dat"] = np.float64(1.0)
    weights["cs_a_dat"] = np.float64(1.0)
    weights["cs_c_dat"] = np.float64(1.0)

    input_params["weights"] = weights

    nn = initialize_nn(args=args, input_params=input_params)
    elapsed, unweighted_loss = do_training(input_params=input_params, nn=nn)
    return elapsed, unweighted_loss


def computeError_optim(trial, dataFolder):
    modelFolder = "Model_" + str(trial.id)
    if args.simpleModel:
        from spm_simpler import makeParams
    else:
        from spm import makeParams
    params = makeParams()
    nn, data_dict, var_dict, params_dict = init_error(
        modelFolder, dataFolder, params
    )
    pred_dict = pinn_pred(nn, var_dict, params_dict)
    globalError, _ = computeError(data_dict, pred_dict, debug=False)
    print(f"GlobalError = {globalError:.4f}")
    return globalError


# parameters = [
#    sherpa.Discrete(name="NEURONS_NUM", range=[10, 100]),
#    sherpa.Discrete(name="LAYERS_T_NUM", range=[1, 8]),
#    sherpa.Discrete(name="LAYERS_TR_NUM", range=[1, 8]),
#    sherpa.Discrete(name="LAYERS_T_VAR_NUM", range=[0, 8]),
#    sherpa.Discrete(name="LAYERS_TR_VAR_NUM", range=[0, 8]),
#    sherpa.Discrete(name="NUM_RES_BLOCKS", range=[1, 8]),
#    sherpa.Discrete(name="NUM_RES_BLOCK_LAYERS", range=[1, 4]),
#    sherpa.Discrete(name="NUM_RES_BLOCK_UNITS", range=[4, 100]),
#    sherpa.Choice(name="ACTIVATION", range=["tanh", "swish", "gelu"]),
# ]


parameters = [
    sherpa.Continuous(name="w_cs_int", range=[1, 100]),
    sherpa.Continuous(name="w_phi_int", range=[1, 100]),
    sherpa.Continuous(name="w_cs_rmin_bound", range=[1, 100]),
    sherpa.Continuous(name="w_cs_rmax_bound", range=[1, 100]),
]


alg = sherpa.algorithms.RandomSearch(max_num_trials=200)
# alg = sherpa.algorithms.GPyOpt(max_num_trials=200)
study = sherpa.Study(
    parameters=parameters,
    algorithm=alg,
    disable_dashboard=True,
    lower_is_better=True,
)

args = argument.initArg()
input_params = initialize_params(args)

import pickle

record = []
record_ckpt = []
skipCounter = 0
if args.restart_from_checkpoint:
    try:
        print("\n\nINFO: READ CHECKPOINT")
        with open("record.pkl", "rb") as fp:
            record_ckpt = pickle.load(fp)
        for rec_ckpt in record_ckpt:
            tmp = {}
            trial_ckpt = rec_ckpt["trial"]
            obj_ckpt = rec_ckpt["objective"]
            try:
                time_ckpt = rec_ckpt["time"]
                loss_ckpt = rec_ckpt["loss"]
                record.append(
                    {
                        "trial": trial_ckpt,
                        "objective": obj_ckpt,
                        "loss": loss_ckpt,
                        "time": time_ckpt,
                    }
                )
            except:
                record.append({"trial": trial_ckpt, "objective": obj_ckpt})

            print(f"\tADD OBS {rec_ckpt['trial'].id}")
            study.add_observation(
                trial=rec_ckpt["trial"], objective=rec_ckpt["objective"]
            )
    except FileNotFoundError:
        print("\n\nINFO: NO CHECKPOINT LOADED")


print("\n\nINFO: START OPTIM")
for trial in study:
    trial.id += len(record_ckpt)
    print(f"\n\nTRIAL {trial.id}")
    print(trial.parameters)
    elapsed, unweighted_loss = main_optim_lossWeights(
        trial, args, input_params
    )
    globalError = computeError_optim(trial, args.dataFolder)
    record.append(
        {
            "trial": trial,
            "objective": globalError,
            "loss": unweighted_loss,
            "time": elapsed,
        }
    )
    if args.save_ckpt:
        with open("record.pkl", "wb") as fp:
            pickle.dump(record, fp)
    study.add_observation(trial=trial, objective=globalError)

study.finalize(trial)

best_param = study.get_best_result()
with open("best_param.pkl", "wb") as fp:
    pickle.dump(best_param, fp)
print(best_param)
