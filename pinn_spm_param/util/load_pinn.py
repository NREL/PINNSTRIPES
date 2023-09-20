import json
import os
from importlib.machinery import SourceFileLoader

import argument
import numpy as np

# Read command line arguments
args = argument.initArg()


def reload(utilFolder, localUtilFolder, params_loaded, checkRescale=False):
    # Reload correct modules
    print("####\t" + str(localUtilFolder))
    _losses = SourceFileLoader(
        "_losses", os.path.join(localUtilFolder, "_losses.py")
    ).load_module()
    _rescale = SourceFileLoader(
        "_rescale", os.path.join(localUtilFolder, "_rescale.py")
    ).load_module()
    uocp_cs = SourceFileLoader(
        "uocp_cs", os.path.join(localUtilFolder, "uocp_cs.py")
    ).load_module()
    thermo = SourceFileLoader(
        "thermo", os.path.join(localUtilFolder, "thermo.py")
    ).load_module()
    if args.simpleModel:
        spm_simpler = SourceFileLoader(
            "spm_simpler", os.path.join(localUtilFolder, "spm_simpler.py")
        ).load_module()
        from spm_simpler import makeParams
    else:
        spm = SourceFileLoader(
            "spm", os.path.join(localUtilFolder, "spm.py")
        ).load_module()
        from spm import makeParams
    myNN = SourceFileLoader(
        "myNN", os.path.join(localUtilFolder, "myNN.py")
    ).load_module()
    init_pinn = SourceFileLoader(
        "init_pinn", os.path.join(localUtilFolder, "init_pinn.py")
    ).load_module()


def load_model(
    utilFolder, modelFolder, localUtilFolder, loadDep=False, checkRescale=False
):
    # Load correct modules
    _losses = SourceFileLoader(
        "_losses", os.path.join(utilFolder, "_losses.py")
    ).load_module()
    _rescale = SourceFileLoader(
        "_rescale", os.path.join(utilFolder, "_rescale.py")
    ).load_module()
    uocp_cs = SourceFileLoader(
        "uocp_cs", os.path.join(utilFolder, "uocp_cs.py")
    ).load_module()
    thermo = SourceFileLoader(
        "thermo", os.path.join(utilFolder, "thermo.py")
    ).load_module()
    if args.simpleModel:
        spm_simpler = SourceFileLoader(
            "spm_simpler", os.path.join(utilFolder, "spm_simpler.py")
        ).load_module()
        from spm_simpler import makeParams
    else:
        spm = SourceFileLoader(
            "spm", os.path.join(utilFolder, "spm.py")
        ).load_module()
        from spm import makeParams
    myNN = SourceFileLoader(
        "myNN", os.path.join(utilFolder, "myNN.py")
    ).load_module()
    init_pinn = SourceFileLoader(
        "init_pinn", os.path.join(localUtilFolder, "init_pinn.py")
    ).load_module()

    params_loaded = makeParams()

    from myNN import myNN

    # Load config
    with open(os.path.join(modelFolder, "config.json")) as json_file:
        configDict = json.load(json_file)
    nn = init_pinn.initialize_nn_from_params_config(params_loaded, configDict)

    # Load weights
    nn = init_pinn.safe_load(nn, os.path.join(modelFolder, "best.h5"))

    # Reload
    reload(
        utilFolder, localUtilFolder, params_loaded, checkRescale=checkRescale
    )

    return nn
