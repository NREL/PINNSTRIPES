import os
from pathlib import Path

import black
import numpy as np
import pandas as pd
import tf_lineInterp as tfli
from myProgressBar import printProgressBar
from scipy import odr
from scipy.optimize import minimize


def polynomial(p, xd):
    poly = np.poly1d(p)
    return poly(xd)


def constraint_1st_der(p, xd):
    poly = np.poly1d(p)
    df = np.polyder(poly, m=1)
    return np.clip(df(xd), a_min=0, a_max=None)


def constraint_2nd_der(p, xd):
    poly = np.poly1d(p)
    df = np.polyder(poly, m=2)
    return df(xd) ** 2


def objective(p, xd, yd):
    return ((polynomial(p, xd) - yd) ** 2).sum() + 1e1 * constraint_1st_der(
        p, xd
    ).sum()


def polyOpt(order, xd, yd):
    poly_model = odr.polynomial(order)
    data = odr.Data(xd, yd)
    odr_obj = odr.ODR(data, poly_model)
    output = odr_obj.run()
    poly = np.poly1d(output.beta[::-1])
    res = minimize(
        objective, x0=output.beta[::-1], args=(xd, yd), method="SLSQP"
    )
    return res


def iterativeOpt(order_list, xd, yd):
    error = []
    printProgressBar(
        0,
        len(order_list) - 1,
        prefix=f"Poly Order = {0} / {len(order_list)-1}",
        suffix="Complete",
        length=20,
    )
    for order in order_list:
        res = polyOpt(order, xd, yd)
        pars = res.x
        poly = np.poly1d(pars)
        error.append(np.sum(abs(poly(xd) - yd)))
        printProgressBar(
            order,
            len(order_list) - 1,
            prefix=f"Poly Order = {order} / {len(order_list)-1}",
            suffix="Complete",
            length=20,
        )
    bestOrd = order_list[np.argmin(np.array(error))]
    return bestOrd


def gen_uocp_poly():
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
    slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
    y += list(y[-1] + (np.linspace(x[-1], 1, 20) - x[-1]) * slope)
    x += list(np.linspace(x[-1], 1, 20))
    slope = (y[1] - y[0]) / (x[1] - x[0])
    xd = x
    yd = y

    print("INFO: Generate UOCP An, with polynomial")
    bestOrd = iterativeOpt(list(range(30)), xd, yd)
    res = polyOpt(bestOrd, xd, yd)
    pars = res.x
    poly = np.poly1d(pars)

    tfli.generateTFPoly(
        list(pars), funcname="uocp_a_fun_x", filename=file_to_write, mode="w+"
    )
    tfli.generateComsolPoly(
        list(pars), filename="comsol_uocp", funcname="uocp_an", mode="w+"
    )

    refFolder = "../../Data/stateEqRef"
    uocp_c_ref_file = os.path.join(refFolder, "caEeq.csv")
    uocp_c_ref_data = pd.read_csv(uocp_c_ref_file).to_numpy()
    x_ref = uocp_c_ref_data[:, 0]
    uocp_c_ref = uocp_c_ref_data[:, 1]
    x = list(x_ref)
    y = list(uocp_c_ref)
    slope = (y[1] - y[0]) / (x[1] - x[0])
    ybeg = list(y[0] + (np.linspace(0, 0.32, 100) - x[0]) * slope)
    xbeg = list(np.linspace(0, 0.32, 100))
    xd = xbeg + x
    yd = ybeg + y

    print("INFO: Generate UOCP Ca, with polynomial")
    bestOrd = iterativeOpt(list(range(30)), xd, yd)
    res = polyOpt(bestOrd, xd, yd)
    pars = res.x
    poly = np.poly1d(pars)

    tfli.generateTFPoly(
        list(pars), funcname="uocp_c_fun_x", filename=file_to_write, mode="a+"
    )
    tfli.generateComsolPoly(
        list(pars), filename="comsol_uocp", funcname="uocp_ca", mode="a+"
    )

    # Format generated code
    mode = black.FileMode(line_length=79)
    fast = False
    success = black.format_file_in_place(
        Path(file_to_write),
        fast=False,
        mode=mode,
        write_back=black.WriteBack.YES,
    )


if __name__ == "__main__":
    gen_uocp_poly()
