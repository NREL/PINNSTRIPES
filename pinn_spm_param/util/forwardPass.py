import json
import sys

import numpy as np

sys.path.append("../util")
import argument
import tensorflow as tf
from keras import layers, regularizers
from myNN import *
from keras.backend import set_floatx

set_floatx("float64")

from init_pinn import initialize_nn_from_params_config


def from_param_list_to_str(params_list, params_name=None):
    param_string = ""
    if params_list is not None:
        if isinstance(params_list[0], str):
            params_list_val = [float(val) for val in params_list]
        else:
            params_list_val = params_list
        if params_name is None:
            for paramval in params_list_val:
                param_string += "_"
                param_string += f"{paramval:.3g}"
        else:
            for paramval, name in zip(params_list_val, params_name):
                param_string += f"_{name}_"
                param_string += f"{paramval:.3g}"
    return param_string


def rescale_param_list(nn, param_list):
    return [
        nn.rescale_param(param_list[i], i) for i in range(len(nn.resc_params))
    ]


def var_from_x(x_arr, dummyVal=np.float64(0.0)):
    var_simp = [None] * x_arr.shape[1]
    var_full = [None] * max(x_arr.shape[1], 2)
    var_simp = [x_arr[:, i] for i in range(x_arr.shape[1])]
    var_full[0] = var_simp[0]
    if x_arr.shape[1] == 2:
        var_full[1] = var_simp[1]
    else:
        var_full[1] = dummyVal * np.ones(x_arr.shape[0])
    return var_simp, var_full


def rescale_var_list(nn, var_list, dummyVal=np.float64(0.0)):
    var_list_resc = [None] * len(var_list)
    var_list_resc_full = [None] * max(len(var_list), 2)
    var_list_resc[nn.ind_t] = var_list[nn.ind_t] / nn.params["rescale_T"]
    var_list_resc_full[nn.ind_t] = var_list_resc[nn.ind_t]
    if len(var_list) == 2:
        var_list_resc[nn.ind_r] = var_list[nn.ind_r] / nn.params["rescale_R"]
        var_list_resc_full[nn.ind_r] = var_list_resc[nn.ind_r]
    else:
        var_list_resc_full[nn.ind_r] = (
            dummyVal
            * np.ones(var_list_resc_full[nn.ind_t].shape)
            / nn.params["rescale_R"]
        )
    return var_list_resc, var_list_resc_full


def make_data_dict(dataFolder, param_list=None):
    param_string = from_param_list_to_str(param_list)

    data_dict = {}

    data_phie = np.load(
        os.path.join(dataFolder, f"data_phie{param_string}.npz")
    )
    data_dict["var_phie"] = data_phie["x_test"].astype("float64")
    data_dict["phie"] = data_phie["y_test"].astype("float64")
    data_dict["params_phie"] = data_phie["x_params_test"].astype("float64")
    data_phis_c = np.load(
        os.path.join(dataFolder, f"data_phis_c{param_string}.npz")
    )
    data_dict["var_phis_c"] = data_phis_c["x_test"].astype("float64")
    data_dict["phis_c"] = data_phis_c["y_test"].astype("float64")
    data_dict["params_phis_c"] = data_phis_c["x_params_test"].astype("float64")
    data_cs_a = np.load(
        os.path.join(dataFolder, f"data_cs_a{param_string}.npz")
    )
    data_dict["var_cs_a"] = data_cs_a["x_test"].astype("float64")
    data_dict["cs_a"] = data_cs_a["y_test"].astype("float64")
    data_dict["params_cs_a"] = data_cs_a["x_params_test"].astype("float64")
    data_cs_c = np.load(
        os.path.join(dataFolder, f"data_cs_c{param_string}.npz")
    )
    data_dict["var_cs_c"] = data_cs_c["x_test"].astype("float64")
    data_dict["cs_c"] = data_cs_c["y_test"].astype("float64")
    data_dict["params_cs_c"] = data_cs_c["x_params_test"].astype("float64")

    return data_dict


def make_data_dict_struct(dataFolder, param_list=None):
    param_string = from_param_list_to_str(param_list)
    sol = np.load(os.path.join(dataFolder, f"solution{param_string}.npz"))

    data_dict = {}
    data_dict["phie"] = sol["phie"]
    data_dict["phie_t"] = sol["t"]
    data_dict["phis_c"] = sol["phis_c"]
    data_dict["phis_c_t"] = sol["t"]
    data_dict["cs_a"] = sol["cs_a"]
    data_dict["cs_a_t"] = sol["t"]
    data_dict["cs_a_r"] = sol["r_a"]
    data_dict["cs_c"] = sol["cs_c"]
    data_dict["cs_c_t"] = sol["t"]
    data_dict["cs_c_r"] = sol["r_c"]

    return data_dict


def make_var_params_from_data(nn, data_dict):
    var_dict = {}
    params_dict = {}

    nparams = len(nn.resc_params)
    params_dict["phie_unr"] = [
        data_dict["params_phie"][:, i] for i in range(nparams)
    ]
    params_dict["phie"] = rescale_param_list(nn, params_dict["phie_unr"])
    var_dict["phie_unr"], var_dict["phie_unr_full"] = var_from_x(
        data_dict["var_phie"]
    )
    var_dict["phie"], var_dict["phie_full"] = rescale_var_list(
        nn, var_dict["phie_unr"]
    )
    params_dict["phis_c_unr"] = [
        data_dict["params_phis_c"][:, i] for i in range(nparams)
    ]
    params_dict["phis_c"] = rescale_param_list(nn, params_dict["phis_c_unr"])
    var_dict["phis_c_unr"], var_dict["phis_c_unr_full"] = var_from_x(
        data_dict["var_phis_c"], nn.params["Rs_c"]
    )
    var_dict["phis_c"], var_dict["phis_c_full"] = rescale_var_list(
        nn, var_dict["phis_c_unr"]
    )
    params_dict["cs_a_unr"] = [
        data_dict["params_cs_a"][:, i] for i in range(nparams)
    ]
    params_dict["cs_a"] = rescale_param_list(nn, params_dict["cs_a_unr"])
    var_dict["cs_a_unr"], var_dict["cs_a_unr_full"] = var_from_x(
        data_dict["var_cs_a"], nn.params["Rs_a"]
    )
    var_dict["cs_a"], var_dict["cs_a_full"] = rescale_var_list(
        nn, var_dict["cs_a_unr"]
    )
    params_dict["cs_c_unr"] = [
        data_dict["params_cs_c"][:, i] for i in range(nparams)
    ]
    params_dict["cs_c"] = rescale_param_list(nn, params_dict["cs_c_unr"])
    var_dict["cs_c_unr"], var_dict["cs_c_unr_full"] = var_from_x(
        data_dict["var_cs_c"], nn.params["Rs_c"]
    )
    var_dict["cs_c"], var_dict["cs_c_full"] = rescale_var_list(
        nn, var_dict["cs_c_unr"]
    )

    return var_dict, params_dict


def pinn_pred(nn, var_dict, params_dict):
    pred_dict = {}

    out = nn.model(var_dict["phie_full"] + params_dict["phie"])
    phie_nonrescaled = out[nn.ind_phie]
    phie_rescaled = nn.rescalePhie(
        phie_nonrescaled, *var_dict["phie_unr"], *params_dict["phie_unr"]
    )
    pred_dict["phie"] = phie_rescaled

    out = nn.model(var_dict["phis_c_full"] + params_dict["phis_c"])
    phis_c_nonrescaled = out[nn.ind_phis_c]
    phis_c_rescaled = nn.rescalePhis_c(
        phis_c_nonrescaled, *var_dict["phis_c_unr"], *params_dict["phis_c_unr"]
    )
    pred_dict["phis_c"] = phis_c_rescaled

    out = nn.model(var_dict["cs_a_full"] + params_dict["cs_a"])
    cs_a_nonrescaled = out[nn.ind_cs_a]
    cs_a_rescaled = nn.rescaleCs_a(
        cs_a_nonrescaled, *var_dict["cs_a_unr_full"], *params_dict["cs_a_unr"]
    )
    pred_dict["cs_a"] = cs_a_rescaled

    out = nn.model(var_dict["cs_c_full"] + params_dict["cs_c"])
    cs_c_nonrescaled = out[nn.ind_cs_c]
    cs_c_rescaled = nn.rescaleCs_c(
        cs_c_nonrescaled, *var_dict["cs_c_unr_full"], *params_dict["cs_c_unr"]
    )
    pred_dict["cs_c"] = cs_c_rescaled

    return pred_dict


def pinn_pred_phis_c(nn, var_dict, params_dict):
    pred_dict = {}

    out = nn.model(var_dict["phis_c_full"] + params_dict["phis_c"])
    phis_c_nonrescaled = out[nn.ind_phis_c]
    phis_c_rescaled = nn.rescalePhis_c(
        phis_c_nonrescaled, *var_dict["phis_c_unr"], *params_dict["phis_c_unr"]
    )
    pred_dict["phis_c"] = phis_c_rescaled

    return pred_dict


def pinn_pred_struct(nn, params_list):
    n_t = 32
    n_r = 32
    n_par = 1

    tmin = nn.params["tmin"]
    tmax = nn.params["tmax"]
    rmin = nn.params["rmin"]
    rmax_a = nn.params["Rs_a"]
    rmax_s = max(nn.params["Rs_a"], nn.params["Rs_c"])
    rmax_c = nn.params["Rs_c"]

    t_test = np.reshape(np.linspace(tmin, tmax, n_t), (n_t, 1, 1, 1))
    t_test_a = np.reshape(np.linspace(tmin, tmax, n_t), (n_t, 1, 1, 1))
    r_test_a = np.reshape(np.linspace(rmin, rmax_a, n_r), (1, n_r, 1, 1))

    t_a = np.reshape(
        np.repeat(
            np.repeat(np.repeat(t_test_a, n_r, axis=1), n_par, axis=2),
            n_par,
            axis=3,
        ),
        (n_r * n_t * n_par * n_par, 1),
    ).astype("float64")
    r_a = np.reshape(
        np.repeat(
            np.repeat(np.repeat(r_test_a, n_t, axis=0), n_par, axis=2),
            n_par,
            axis=3,
        ),
        (n_r * n_t * n_par * n_par, 1),
    ).astype("float64")
    r_surf_a = np.reshape(
        np.linspace(rmax_a, rmax_a, n_t * n_r * n_par * n_par),
        (n_t * n_r * n_par * n_par, 1),
    ).astype("float64")

    t_test_c = np.reshape(np.linspace(tmin, tmax, n_t), (n_t, 1, 1, 1))
    r_test_c = np.reshape(np.linspace(rmin, rmax_c, n_r), (1, n_r, 1, 1))
    t_c = np.reshape(
        np.repeat(
            np.repeat(np.repeat(t_test_c, n_r, axis=1), n_par, axis=2),
            n_par,
            axis=3,
        ),
        (n_r * n_t * n_par * n_par, 1),
    ).astype("float64")
    r_c = np.reshape(
        np.repeat(
            np.repeat(np.repeat(r_test_c, n_t, axis=0), n_par, axis=2),
            n_par,
            axis=3,
        ),
        (n_r * n_t * n_par * n_par, 1),
    ).astype("float64")
    r_surf_c = np.reshape(
        np.linspace(rmax_c, rmax_c, n_t * n_r * n_par * n_par),
        (n_t * n_r * n_par * n_par, 1),
    ).astype("float64")
    params_r_unr = [
        np.ones((n_t * n_r, 1)) * entry
        for ientry, entry in enumerate(params_list)
    ]
    params_r = [
        np.ones((n_t * n_r, 1)) * nn.rescale_param(entry, ientry)
        for ientry, entry in enumerate(params_list)
    ]

    data_dict = {}
    data_dict["var_phie"] = t_a
    data_dict["params_phie"] = np.hstack(params_r_unr)

    data_dict["var_phis_c"] = t_c
    data_dict["params_phis_c"] = np.hstack(params_r_unr)

    data_dict["var_cs_a"] = np.hstack((t_a, r_a))
    data_dict["params_cs_a"] = np.hstack(params_r_unr)

    data_dict["var_cs_c"] = np.hstack((t_c, r_c))
    data_dict["params_cs_c"] = np.hstack(params_r_unr)

    var_dict, params_dict = make_var_params_from_data(nn, data_dict)
    pred_dict = pinn_pred(nn, var_dict, params_dict)

    phie = pred_dict["phie"]
    phis_c = pred_dict["phis_c"]
    cs_a = pred_dict["cs_a"]
    cs_c = pred_dict["cs_c"]

    return {
        "phie": np.reshape(phie, (n_t, n_r, n_par, n_par)),
        "phis_c": np.reshape(phis_c, (n_t, n_r, n_par, n_par)),
        "cs_a": np.reshape(cs_a, (n_t, n_r, n_par, n_par)),
        "cs_c": np.reshape(cs_c, (n_t, n_r, n_par, n_par)),
        "t_test": t_test,
        "r_test_a": r_test_a,
        "r_test_c": r_test_c,
        "tmax": tmax,
    }
