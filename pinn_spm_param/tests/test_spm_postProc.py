import sys

sys.path.append("../")
sys.path.append("../util")
sys.path.append("../postProcess")
from main import *


def test_plot_data():
    from plotData import plot_pde_data

    args.dataFolder = "../integration_spm"
    args.params_list = ["0.5", "1.0"]
    plot_pde_data(args)


def test_corr_plot():
    from plotCorrelationPINNvsData import corr_plot

    args.modelFolder = "Model_1"
    args.params_list = ["0.5", "1.0"]
    corr_plot(args)


def test_plot_pinn_result():
    from plotPINNResult import plot_pinn_result

    args.modelFolder = "Model_1"
    args.params_list = ["0.5", "1.0"]
    plot_pinn_result(args)


def test_plot_res_var():
    from plotResidualVariation import plot_res_var

    args.modelFolder = "Model_1"
    args.logFolder = "Log_1"
    plot_res_var(args)


def test_comp_err():
    from computeError import computeError, init_error
    from forwardPass import pinn_pred

    args.modelFolder = "Model_1"
    args.dataFolder = "../integration_spm"
    args.params_list = ["0.5", "1.0"]
    if args.simpleModel:
        from spm_simpler import makeParams
    else:
        from spm import makeParams
    params = makeParams()
    nn, data_dict, var_dict, params_dict = init_error(
        args.modelFolder, args.dataFolder, params, params_list=args.params_list
    )
    pred_dict = pinn_pred(nn, var_dict, params_dict)
    globalError, _ = computeError(data_dict, pred_dict)
    print(globalError)


if __name__ == "__main__":
    test_plot_data()
    test_corr_plot()
    test_plot_pinn_result()
    test_plot_res_var()
    test_comp_err()
    pass
