import sys
import time

import numpy as np
from prettyPlot.progressBar import print_progress_bar


def get_r_domain(n_r, params):
    r_a = np.linspace(0, params["Rs_a"], n_r)
    dR_a = params["Rs_a"] / (n_r - 1)
    r_c = np.linspace(0, params["Rs_c"], n_r)
    dR_c = params["Rs_c"] / (n_r - 1)
    return {"n_r": n_r, "r_a": r_a, "dR_a": dR_a, "r_c": r_c, "dR_c": dR_c}


def get_t_domain(n_t, params):
    t = np.linspace(0, params["tmax"], n_t)
    dt = params["tmax"] / (n_t - 1)
    return {"t": t, "dt": dt, "n_t": n_t}


def get_nt_from_dt(dt, params):
    return int(params["tmax"] // dt)


def get_expl_nt(r_dom, params, deg_ds_c):
    """
    Choose n_t to satisfy the diffusive CFL constraint
    """
    r_a = r_dom["r_a"]
    dR_a = r_dom["dR_a"]
    r_c = r_dom["r_c"]
    dR_c = r_dom["dR_c"]
    mindR = min(dR_a, dR_c)
    Ds_ave = 0.5 * (params["D_s_a"](params["T"], params["R"])) + 0.5 * (
        params["D_s_c"](
            params["cs_c0"],
            params["T"],
            params["R"],
            params["cscamax"],
            deg_ds_c,
        )
    )
    dt_target = mindR**2 / (2 * Ds_ave)
    n_t = int(2 * params["tmax"] // dt_target)

    return n_t


def make_sim_config(t_dom, r_dom):
    config = {**t_dom, **r_dom}
    return config


def init_arrays(n_t, n_r):
    phie = np.zeros(n_t)
    phis_c = np.zeros(n_t)
    cs_a = np.zeros((n_t, n_r))
    cs_c = np.zeros((n_t, n_r))
    Ds_a = np.zeros(n_r)
    Ds_c = np.zeros(n_r)
    rhs_a = np.zeros(n_r)
    rhs_c = np.zeros(n_r)
    A = np.zeros((n_r, n_r))
    B_a = np.zeros(n_r)
    B_c = np.zeros(n_r)

    return {
        "ce": 0,
        "phis_a": 0,
        "phie": phie,
        "phis_c": phis_c,
        "cs_a": cs_a,
        "cs_c": cs_c,
        "Ds_a": Ds_a,
        "Ds_c": Ds_c,
        "rhs_a": rhs_a,
        "rhs_c": rhs_c,
        "A": A,
        "B_a": B_a,
        "B_c": B_c,
    }


def tridiag(ds, dt, dr):
    a = 1 + 2 * ds * dt / (dr**2)
    b = -ds * dt / (dr**2)
    bup = b[:-1]
    bdown = b[1:]
    main_mat = np.diag(a, 0) + np.diag(bdown, -1) + np.diag(bup, 1)
    main_mat[0, :] = 0
    main_mat[-1, :] = 0
    main_mat[0, 0] = -1 / dr
    main_mat[0, 1] = 1 / dr
    main_mat[-1, -1] = 1 / dr
    main_mat[-1, -2] = -1 / dr
    return main_mat


def rhs(dt, r, ddr_cs, ds, ddDs_cs, cs, bound_grad):
    rhs_col = np.zeros(len(r))
    rhs_col = (
        dt
        * (np.float64(2.0) / np.clip(r, a_min=1e-12, a_max=None))
        * ddr_cs
        * ds
    )
    rhs_col += dt * ddr_cs**2 * ddDs_cs
    rhs_col += cs
    rhs_col[0] = 0
    rhs_col[-1] = bound_grad
    return rhs_col


def init_sol(n_t, n_r, params, deg_i0_a, deg_ds_c):
    sol = init_arrays(n_t, n_r)
    sol["ce"] = params["ce0"]
    sol["phis_a"] = 0
    sol["cs_a"][0, :] = params["cs_a0"]
    sol["cs_c"][0, :] = params["cs_c0"]
    # j_a = -params["I_discharge"] / (params["A_a"] * params["F"])
    # j_c = params["I_discharge"] / (params["A_c"] * params["F"])
    sol["j_a"] = params["j_a"]
    sol["j_c"] = params["j_c"]

    Uocp_a = params["Uocp_a"](params["cs_a0"], params["csanmax"])
    i0_a = params["i0_a"](
        params["cs_a0"],
        sol["ce"],
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        deg_i0_a,
    )
    sol["phie"][0] = params["phie0"](
        i0_a,
        sol["j_a"],
        params["F"],
        params["R"],
        params["T"],
        Uocp_a,
    )
    Uocp_c = params["Uocp_c"](params["cs_c0"], params["cscamax"])
    i0_c = params["i0_c"](
        params["cs_c0"],
        params["ce0"],
        params["T"],
        params["alpha_c"],
        params["cscamax"],
        params["R"],
    )
    sol["phis_c"][0] = params["phis_c0"](
        i0_a,
        sol["j_a"],
        params["F"],
        params["R"],
        params["T"],
        Uocp_a,
        sol["j_c"],
        i0_c,
        Uocp_c,
    )
    return sol


def integration(
    sol,
    config,
    params,
    deg_i0_a,
    deg_ds_c,
    explicit=False,
    verbose=False,
    LINEARIZE_J=True,
    EXACT_GRAD_DS_CS=False,
    GRAD_STEP=0.1,
):
    n_t = config["n_t"]
    dt = config["dt"]
    n_r = config["n_r"]
    r_a = config["r_a"]
    dR_a = config["dR_a"]
    r_c = config["r_c"]
    dR_c = config["dR_c"]

    if verbose:
        print_progress_bar(
            0,
            n_t,
            prefix=f"Step= {0} / {n_t}",
            suffix="Complete",
            length=20,
        )
    for i_t in range(1, n_t):
        # for i_t in range(1,2):
        # GET PHIE: -I/A = ja = (2 i0/F) * sinh ( 0.5 * (F/RT) (-phie - Uocp_a))
        cse_a = sol["cs_a"][i_t - 1, -1]
        i0_a = params["i0_a"](
            cse_a,
            sol["ce"],
            params["T"],
            params["alpha_a"],
            params["csanmax"],
            params["R"],
            deg_i0_a,
        )
        Uocp_a = params["Uocp_a"](cse_a, params["csanmax"])
        if not LINEARIZE_J:
            sol["phie"][i_t] = (
                -(np.float64(2.0) * params["R"] * params["T"] / params["F"])
                * np.arcsinh(
                    sol["j_a"] * params["F"] / (np.float64(2.0) * i0_a)
                )
                - Uocp_a
            )
        else:
            sol["phie"][i_t] = (
                -sol["j_a"]
                * (params["F"] / i0_a)
                * (params["R"] * params["T"] / params["F"])
                - Uocp_a
            )

        # GET PHIS_c: I/A = jc = (2 i0/F) * sinh ( 0.5 (F/RT) (phis_c -phie - Uocp_c))
        cse_c = sol["cs_c"][i_t - 1, -1]
        i0_c = params["i0_c"](
            cse_c,
            sol["ce"],
            params["T"],
            params["alpha_c"],
            params["cscamax"],
            params["R"],
        )
        Uocp_c = params["Uocp_c"](cse_c, params["cscamax"])
        if not LINEARIZE_J:
            sol["phis_c"][i_t] = (
                (np.float64(2.0) * params["R"] * params["T"] / params["F"])
                * np.arcsinh(
                    sol["j_c"] * params["F"] / (np.float64(2.0) * i0_c)
                )
                + Uocp_c
                + sol["phie"][i_t]
            )
        else:
            sol["phis_c"][i_t] = (
                sol["j_c"]
                * (params["F"] / i0_c)
                * (params["R"] * params["T"] / params["F"])
                + Uocp_c
                + sol["phie"][i_t]
            )

        # GET cs_a
        if not explicit:
            sol["Ds_a"][:] = params["D_s_a"](params["T"], params["R"])
            if EXACT_GRAD_DS_CS:
                gradDs_a_cs_a = np.array(
                    grad_ds_a_cs_a(params["T"], params["R"])
                )
            else:
                gradDs_a_cs_a = np.zeros(len(sol["Ds_a"]))

            ddr_csa = np.gradient(
                sol["cs_a"][i_t - 1, :], r_a, axis=0, edge_order=2
            )
            ddr_csa[0] = 0
            ddr_csa[-1] = -sol["j_a"] / sol["Ds_a"][-1]

            A_a = tridiag(sol["Ds_a"], dt, dR_a)
            B_a = rhs(
                dt=dt,
                r=r_a,
                ddr_cs=ddr_csa,
                ds=sol["Ds_a"],
                ddDs_cs=gradDs_a_cs_a,
                cs=sol["cs_a"][i_t - 1, :],
                bound_grad=-sol["j_a"] / sol["Ds_a"][-1],
            )

            cs_a_tmp = np.linalg.solve(A_a, B_a)
            sol["cs_a"][i_t, :] = np.clip(
                cs_a_tmp, a_min=0.0, a_max=params["csanmax"]
            )

            # GET cs_c
            sol["Ds_c"][:] = params["D_s_c"](
                sol["cs_c"][i_t - 1, :],
                params["T"],
                params["R"],
                params["cscamax"],
                deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape),
            )
            if EXACT_GRAD_DS_CS:
                gradDs_c_cs_c = grad_ds_c_cs_c(
                    sol["cs_c"][i_t - 1, :],
                    params["T"],
                    params["R"],
                    params["cscamax"],
                )
            else:
                Ds_c_tmp1 = params["D_s_c"](
                    np.clip(
                        sol["cs_c"][i_t - 1, :]
                        + np.ones(sol["cs_c"].shape[1]) * GRAD_STEP,
                        a_min=0,
                        a_max=params["cscamax"],
                    ),
                    params["T"],
                    params["R"],
                    params["cscamax"],
                    deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape),
                )
                Ds_c_tmp2 = params["D_s_c"](
                    np.clip(
                        sol["cs_c"][i_t - 1, :]
                        - np.ones(sol["cs_c"].shape[1]) * GRAD_STEP,
                        a_min=0,
                        a_max=params["cscamax"],
                    ),
                    params["T"],
                    params["R"],
                    params["cscamax"],
                    deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape),
                )
                gradDs_c_cs_c = (
                    (Ds_c_tmp1 - Ds_c_tmp2) / (2 * GRAD_STEP)
                ).numpy()

            ddr_csc = np.gradient(
                sol["cs_c"][i_t - 1, :], r_c, axis=0, edge_order=2
            )
            ddr_csc[0] = 0
            ddr_csc[-1] = -sol["j_c"] / sol["Ds_c"][-1]

            A_c = tridiag(sol["Ds_c"], dt, dR_c)
            B_c = rhs(
                dt=dt,
                r=r_c,
                ddr_cs=ddr_csc,
                ds=sol["Ds_c"],
                ddDs_cs=gradDs_c_cs_c,
                cs=sol["cs_c"][i_t - 1, :],
                bound_grad=-sol["j_c"] / sol["Ds_c"][-1],
            )

            cs_c_tmp = np.linalg.solve(A_c, B_c)

            sol["cs_c"][i_t, :] = np.clip(
                cs_c_tmp, a_min=0.0, a_max=params["cscamax"]
            )
        elif explicit:
            # GET cs_a
            sol["Ds_a"][:] = params["D_s_a"](params["T"], params["R"])
            ddr_csa = np.gradient(
                sol["cs_a"][i_t - 1, :], r_a, axis=0, edge_order=2
            )
            ddr_csa[0] = 0
            ddr_csa[-1] = -sol["j_a"] / sol["Ds_a"][-1]
            ddr2_csa = np.zeros(n_r)
            ddr2_csa[1 : n_r - 1] = (
                sol["cs_a"][i_t - 1, : n_r - 2]
                - 2 * sol["cs_a"][i_t - 1, 1 : n_r - 1]
                + sol["cs_a"][i_t - 1, 2:n_r]
            ) / dR_a**2
            ddr2_csa[0] = (
                sol["cs_a"][i_t - 1, 0]
                - 2 * sol["cs_a"][i_t - 1, 0]
                + sol["cs_a"][i_t - 1, 1]
            ) / dR_a**2
            ddr2_csa[-1] = (
                sol["cs_a"][i_t - 1, -2]
                - 2 * sol["cs_a"][i_t - 1, -1]
                + sol["cs_a"][i_t - 1, -1]
                + ddr_csa[-1] * dR_a
            ) / dR_a**2
            ddr_Ds = np.gradient(sol["Ds_a"], r_a, axis=0, edge_order=2)
            sol["rhs_a"][1:] = (
                sol["Ds_a"][1:] * ddr2_csa[1:]
                + ddr_Ds[1:] * ddr_csa[1:]
                + 2 * sol["Ds_a"][1:] * ddr_csa[1:] / r_a[1:]
            )
            sol["rhs_a"][0] = 3 * sol["Ds_a"][0] * ddr2_csa[0]
            sol["cs_a"][i_t, :] = np.clip(
                sol["cs_a"][i_t - 1, :] + dt * sol["rhs_a"],
                a_min=0.0,
                a_max=None,
            )
            # GET cs_c
            sol["Ds_c"][:] = params["D_s_c"](
                sol["cs_c"][i_t - 1, :],
                params["T"],
                params["R"],
                params["cscamax"],
                deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape),
            )
            ddr_csc = np.gradient(
                sol["cs_c"][i_t - 1, :], r_c, axis=0, edge_order=2
            )
            ddr_csc[0] = 0
            ddr_csc[-1] = -sol["j_c"] / sol["Ds_c"][-1]
            ddr2_csc = np.zeros(n_r)
            ddr2_csc[1 : n_r - 1] = (
                sol["cs_c"][i_t - 1, : n_r - 2]
                - 2 * sol["cs_c"][i_t - 1, 1 : n_r - 1]
                + sol["cs_c"][i_t - 1, 2:n_r]
            ) / dR_c**2
            ddr2_csc[0] = (
                sol["cs_c"][i_t - 1, 0]
                - 2 * sol["cs_c"][i_t - 1, 0]
                + sol["cs_c"][i_t - 1, 1]
            ) / dR_c**2
            ddr2_csc[-1] = (
                sol["cs_c"][i_t - 1, -2]
                - 2 * sol["cs_c"][i_t - 1, -1]
                + sol["cs_c"][i_t - 1, -1]
                + ddr_csc[-1] * dR_c
            ) / dR_c**2
            ddr_Ds = np.gradient(sol["Ds_c"], r_c, axis=0, edge_order=2)
            sol["rhs_c"][1:] = (
                sol["Ds_c"][1:] * ddr2_csc[1:]
                + ddr_Ds[1:] * ddr_csc[1:]
                + 2 * sol["Ds_c"][1:] * ddr_csc[1:] / r_c[1:]
            )
            sol["rhs_c"][0] = 3 * sol["Ds_c"][0] * ddr2_csc[0]
            sol["cs_c"][i_t, :] = np.clip(
                sol["cs_c"][i_t - 1, :] + dt * sol["rhs_c"],
                a_min=0.0,
                a_max=None,
            )

        if verbose:
            print_progress_bar(
                i_t,
                n_t - 1,
                prefix=f"Step= {i_t} / {n_t}",
                suffix="Complete",
                length=20,
            )


def exec_impl(n_r, dt, params, deg_i0_a, deg_ds_c, verbose=False):
    time_s = time.time()
    r_dom = get_r_domain(n_r, params)
    n_t = get_nt_from_dt(dt, params)
    t_dom = get_t_domain(n_t, params)
    config = make_sim_config(t_dom, r_dom)
    sol = init_sol(n_t, n_r, params, deg_i0_a, deg_ds_c)
    integration(
        sol,
        config,
        params,
        deg_i0_a,
        deg_ds_c,
        explicit=False,
        verbose=verbose,
        LINEARIZE_J=True,
        EXACT_GRAD_DS_CS=False,
    )
    time_e = time.time()
    if verbose:
        print(f"n_r: {n_r}, n_t: {n_t}, time = {time_e-time_s:.2f}s")
    return config, sol


def exec_expl(n_r, params, deg_i0_a, deg_ds_c, verbose=False):
    time_s = time.time()
    r_dom = get_r_domain(n_r, params)
    n_t = get_expl_nt(r_dom, params, deg_ds_c)
    t_dom = get_t_domain(n_t, params)
    config = make_sim_config(t_dom, r_dom)
    sol = init_sol(n_t, n_r, params, deg_i0_a, deg_ds_c)
    integration(
        sol,
        config,
        params,
        deg_i0_a,
        deg_ds_c,
        explicit=True,
        verbose=verbose,
        LINEARIZE_J=True,
        EXACT_GRAD_DS_CS=False,
    )
    time_e = time.time()
    if verbose:
        print(f"n_r: {n_r}, n_t: {n_t}, time = {time_e-time_s:.2f}s")
    return config, sol
