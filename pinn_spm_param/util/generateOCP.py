import os
from pathlib import Path

import black
import numpy as np
import pandas as pd
import tf_lineInterp as tfli

file_to_write = "uocp_cs.py"

refFolder = "../../Data/stateEqRef"
uocp_a_ref_file = os.path.join(refFolder, "anEeq.csv")
uocp_a_ref_data = pd.read_csv(uocp_a_ref_file).to_numpy()
x_ref = uocp_a_ref_data[:, 2]
x_ref = x_ref[~np.isnan(x_ref)]
uocp_a_ref = uocp_a_ref_data[:, 3]
uocp_a_ref = uocp_a_ref[~np.isnan(uocp_a_ref)]


x = list(x_ref)
y = list(uocp_a_ref)
tfli.generateTFSpline(x, y, funcname="uocp_a_fun_x", filename=file_to_write)
tfli.generateComsolSpline(
    x, y, filename="comsol_uocp", funcname="uocp_an", mode="w+"
)

uocp_c_ref_file = os.path.join(refFolder, "caEeq.csv")
uocp_c_ref_data = pd.read_csv(uocp_c_ref_file).to_numpy()
x_ref = uocp_c_ref_data[:, 0]
uocp_c_ref = uocp_c_ref_data[:, 1]

x = list(x_ref)
y = list(uocp_c_ref)
tfli.generateTFSpline(
    x,
    y,
    funcname="uocp_c_fun_x",
    filename=file_to_write,
    mode="a+",
)
tfli.generateComsolSpline(
    x, y, filename="comsol_uocp", funcname="uocp_ca", mode="a+"
)

# Format generated code
mode = black.FileMode(line_length=79)
fast = False
success = black.format_file_in_place(
    Path(file_to_write), fast=False, mode=mode, write_back=black.WriteBack.YES
)
