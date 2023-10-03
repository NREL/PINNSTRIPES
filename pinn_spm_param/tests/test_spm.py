import sys

sys.path.append("../")
sys.path.append("../util")
from main import *

inputFile = "../input"
basicKwargs = {
    "ID": "1",
    "LOAD_MODEL": "None",
    "alpha": "1.0 1.0 0.0 0.0",
    "w_phie_int": "1.0",
    "w_phis_c_int": "1.0",
    "w_cs_a_int": "1.0",
    "w_cs_c_int": "1.0",
    "w_cs_a_rmin_bound": "1.0",
    "w_cs_c_rmin_bound": "1.0",
    "w_cs_a_rmax_bound": "1.0",
    "w_cs_c_rmax_bound": "1.0",
    "w_phie_dat": "1.0",
    "w_phis_c_dat": "1.0",
    "w_cs_a_dat": "1.0",
    "w_cs_c_dat": "1.0",
    "LBFGS": "True",
    "SGD": "True",
    "DYNAMIC_ATTENTION_WEIGHTS": "False",
    "ANNEALING_WEIGHTS": "False",
    "START_WEIGHT_TRAINING_EPOCHS": "1",
    "EPOCHS": "2",
    "INNER_EPOCHS": "0",
    "EPOCHS_LBFGS": "2",
    "EPOCHS_START_LBFGS": "5",
    "BATCH_SIZE_INT": "2",
    "BATCH_SIZE_BOUND": "2",
    "BATCH_SIZE_REG": "0",
    "N_BATCH": "2",
    "NEURONS_NUM": "2",
    "NUM_RES_BLOCKS": "1",
    "NUM_RES_BLOCK_UNITS": "2",
    "NUM_GRAD_PATH_LAYERS": "2",
    "NUM_GRAD_PATH_UNITS": "2",
    "LAYERS_T_NUM": "1",
    "LAYERS_TR_NUM": "1",
    "LAYERS_SPLIT_NUM": "1",
    "NUM_RES_BLOCK_LAYERS": "1",
    "COLLOCATION_MODE": "fixed",
    "GRADUAL_TIME_SGD": "False",
    "GRADUAL_TIME_LBFGS": "False",
    "MERGED": "True",
    "LINEARIZE_J": "True",
}


def initialize_params_test(file, kwargs=None):
    inpt = parseInputFile(file)
    if kwargs is not None:
        for key in kwargs:
            inpt[key] = kwargs[key]
    return initialize_params_from_inpt(inpt)


def basic_train_test(kwargs):
    input_params = initialize_params_test(inputFile, kwargs)
    nn = initialize_nn(args=args, input_params=input_params)
    elapsed, unweighted_loss = do_training_only(
        input_params=input_params, nn=nn
    )
    print(f"Total time {elapsed:.2f}s")
    print(f"Unweighted loss {unweighted_loss}")


def test_sgd_lbfgs():
    kwargs = basicKwargs
    basic_train_test(kwargs)


def test_anneal():
    new_kwargs = {
        "ANNEALING_WEIGHTS": "True",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)


def test_sa():
    new_kwargs = {
        "DYNAMIC_ATTENTION_WEIGHTS": "True",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)


def test_random_col():
    new_kwargs = {
        "COLLOCATION_MODE": "random",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)


def test_split():
    new_kwargs = {
        "MERGED": "False",
        "NUM_GRAD_PATH_LAYERS": "0",
        "NUM_GRAD_PATH_UNITS": "0",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)


def test_fullNL():
    new_kwargs = {
        "LINEARIZE_J": "False",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)


def test_gradualSGD():
    new_kwargs = {
        "GRADUAL_TIME_SGD": "True",
        "EPOCHS": "20",
        "N_BATCH": "1",
        "LBFGS": "False",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)


def test_gradualLBFGS():
    new_kwargs = {
        "GRADUAL_TIME_LBFGS": "True",
        "EPOCHS_LBFGS": "5",
        "SGD": "False",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)


def test_hierarchical():
    # Level 1
    new_kwargs = {
        "alpha": "1.0 1.0 0.0 0.0",
        "ID": "1",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)

    # Level 2
    new_kwargs = {
        "alpha": "1.0 1.0 0.0 0.0",
        "ID": "2",
        "LOCAL_utilFolder": os.path.join(os.getcwd(), "../util"),
        "HNN_utilFolder": os.path.join(os.getcwd(), "../util"),
        "HNN_modelFolder": os.path.join(os.getcwd(), "Model_1"),
        "EPOCHS_LBFGS": "0",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)

    # Level 2 params
    new_kwargs = {
        "alpha": "1.0 1.0 0.0 0.0",
        "ID": "3",
        "LOCAL_utilFolder": os.path.join(os.getcwd(), "../util"),
        "HNN_utilFolder": os.path.join(os.getcwd(), "../util"),
        "HNN_modelFolder": os.path.join(os.getcwd(), "Model_1"),
        "HNN_params": "0.5 1.0",
        "EPOCHS_LBFGS": "0",
    }
    kwargs = {**basicKwargs, **new_kwargs}
    basic_train_test(kwargs)


if __name__ == "__main__":
    test_sgd_lbfgs()
    test_anneal()
    test_sa()
    test_random_col()
    test_split()
    test_fullNL()
    test_gradualSGD()
    test_gradualLBFGS()
    test_hierarchical()
    pass
