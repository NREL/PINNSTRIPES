import argument
import numpy as np
import tensorflow as tf
from conditionalDecorator import conditional_decorator

tf.keras.backend.set_floatx("float64")

# Read command line arguments
args = argument.initArg()

if args.optimized:
    optimized = True
else:
    optimized = False


# @conditional_decorator(tf.function,optimized)
def loss_fn_lbfgs_SA(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_col_weights,
    bound_col_weights,
    data_col_weights,
    reg_col_weights,
    alpha,
):

    # Interior loss
    int_loss = np.float64(0.0)
    for iterm, term in enumerate(interiorTerms):
        int_loss += tf.reduce_mean(tf.square(int_col_weights[iterm] * term))

    # Boundary loss
    bound_loss = np.float64(0.0)
    for iterm, term in enumerate(boundaryTerms):
        bound_loss += tf.reduce_mean(
            tf.square(bound_col_weights[iterm] * term)
        )

    # Data loss
    data_loss = np.float64(0.0)
    for iterm, term in enumerate(dataTerms):
        data_loss += tf.reduce_mean(tf.square(data_col_weights[iterm] * term))

    # Reg loss
    reg_loss = np.float64(0.0)
    for iterm, term in enumerate(regularizationTerms):
        reg_loss += tf.reduce_mean(tf.square(reg_col_weights[iterm] * term))

    global_loss = (
        alpha[0] * int_loss
        + alpha[1] * bound_loss
        + alpha[2] * data_loss
        + alpha[3] * reg_loss
    )
    return (
        global_loss,
        alpha[0] * int_loss,
        alpha[1] * bound_loss,
        alpha[2] * data_loss,
        alpha[3] * reg_loss,
    )


# @conditional_decorator(tf.function,optimized)
def loss_fn_lbfgs(
    interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha
):

    # Interior loss
    int_loss = np.float64(0.0)
    for iterm, term in enumerate(interiorTerms):
        int_loss += tf.reduce_mean(tf.square(term))

    # Boundary loss
    bound_loss = np.float64(0.0)
    for iterm, term in enumerate(boundaryTerms):
        bound_loss += tf.reduce_mean(tf.square(term))

    # Data loss
    data_loss = np.float64(0.0)
    for iterm, term in enumerate(dataTerms):
        data_loss += tf.reduce_mean(tf.square(term))

    # Reg loss
    reg_loss = np.float64(0.0)
    for iterm, term in enumerate(regularizationTerms):
        reg_loss += tf.reduce_mean(tf.square(term))

    global_loss = (
        alpha[0] * int_loss
        + alpha[1] * bound_loss
        + alpha[2] * data_loss
        + alpha[3] * reg_loss
    )

    return (
        global_loss,
        alpha[0] * int_loss,
        alpha[1] * bound_loss,
        alpha[2] * data_loss,
        alpha[3] * reg_loss,
    )


@conditional_decorator(tf.function, optimized)
def loss_fn_dynamicAttention_tensor(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_col_weights,
    bound_col_weights,
    data_col_weights,
    reg_col_weights,
    alpha,
):

    # Interior loss
    int_loss = np.float64(0.0)
    int_loss_unweighted = np.float64(0.0)
    int_loss_unweighted = tf.reduce_mean(tf.square(interiorTerms)) * tf.cast(
        tf.shape(interiorTerms)[0], dtype=tf.float64
    )
    int_loss = tf.reduce_mean(
        tf.square(int_col_weights * interiorTerms)
    ) * tf.cast(tf.shape(interiorTerms)[0], dtype=tf.float64)
    int_loss = int_loss * alpha[0]
    int_loss_unweighted = int_loss_unweighted * alpha[0]

    # Boundary loss
    bound_loss = np.float64(0.0)
    bound_loss_unweighted = np.float64(0.0)
    bound_loss_unweighted = tf.reduce_mean(tf.square(boundaryTerms)) * tf.cast(
        tf.shape(boundaryTerms)[0], dtype=tf.float64
    )
    bound_loss = tf.reduce_mean(
        tf.square(bound_col_weights * boundaryTerms)
    ) * tf.cast(tf.shape(boundaryTerms)[0], dtype=tf.float64)
    bound_loss = bound_loss * alpha[1]
    bound_loss_unweighted = bound_loss_unweighted * alpha[1]

    # Data loss
    data_loss = np.float64(0.0)
    data_loss_unweighted = np.float64(0.0)
    data_loss_unweighted = tf.reduce_mean(tf.square(dataTerms)) * tf.cast(
        tf.shape(dataTerms)[0], dtype=tf.float64
    )
    data_loss = tf.reduce_mean(
        tf.square(data_col_weights * dataTerms)
    ) * tf.cast(tf.shape(dataTerms)[0], dtype=tf.float64)
    data_loss = data_loss * alpha[2]
    data_loss_unweighted = data_loss_unweighted * alpha[2]

    # Regularization loss
    reg_loss = np.float64(0.0)
    reg_loss_unweighted = np.float64(0.0)
    reg_loss_unweighted = tf.reduce_mean(
        tf.square(regularizationTerms)
    ) * tf.cast(tf.shape(regularizationTerms)[0], dtype=tf.float64)
    reg_loss = tf.reduce_mean(
        tf.square(reg_col_weights * regularizationTerms)
    ) * tf.cast(tf.shape(regularizationTerms)[0], dtype=tf.float64)
    reg_loss = reg_loss * alpha[3]
    reg_loss_unweighted = reg_loss_unweighted * alpha[3]

    return (
        int_loss + bound_loss + data_loss + reg_loss,
        int_loss_unweighted
        + bound_loss_unweighted
        + data_loss_unweighted
        + reg_loss_unweighted,
        int_loss,
        bound_loss,
        data_loss,
        reg_loss,
    )


# @conditional_decorator(tf.function,optimized)
def loss_fn(
    interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha
):

    # Interior loss
    int_loss = np.float64(0.0)
    for iterm, term in enumerate(interiorTerms):
        int_loss += tf.reduce_mean(tf.square(term))

    int_loss = int_loss * alpha[0]

    # Boundary loss
    bound_loss = np.float64(0.0)
    for iterm, term in enumerate(boundaryTerms):
        bound_loss += tf.reduce_mean(tf.square(term))
    bound_loss = bound_loss * alpha[1]

    # Data loss
    data_loss = np.float64(0.0)
    for iterm, term in enumerate(dataTerms):
        data_loss += tf.reduce_mean(tf.square(term))
    data_loss = data_loss * alpha[2]

    # Regularization loss
    reg_loss = np.float64(0.0)
    for iterm, term in enumerate(regularizationTerms):
        reg_loss += tf.reduce_mean(tf.square(term))
    reg_loss = reg_loss * alpha[3]

    return (
        int_loss + bound_loss + data_loss + reg_loss,
        int_loss,
        bound_loss,
        data_loss,
        reg_loss,
    )


# L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
def get_loss_and_flat_grad_SA(
    self,
    int_col_pts,
    int_col_params,
    bound_col_pts,
    bound_col_params,
    reg_col_pts,
    reg_col_params,
    int_col_weights,
    bound_col_weights,
    data_col_weights,
    reg_col_weights,
    x_trainList,
    x_params_trainList,
    y_trainList,
    n_batch=1,
    tmax=None,
):
    def loss_and_flat_grad(w):
        accumulatedGradient = 0
        accumulatedLoss = 0
        accumulatedUnweightedLoss = 0
        accumulatedLossInt = 0
        accumulatedLossBound = 0
        accumulatedLossData = 0
        accumulatedLossReg = 0
        batch_size_int = self.batch_size_int_lbfgs
        batch_size_bound = self.batch_size_bound_lbfgs
        batch_size_data = self.batch_size_data_lbfgs
        batch_size_reg = self.batch_size_reg_lbfgs
        for i_batch in range(n_batch):
            int_col_pts_batch = [
                pts[i_batch * batch_size_int : (i_batch + 1) * batch_size_int]
                for pts in int_col_pts
            ]
            int_col_params_batch = [
                pts[i_batch * batch_size_int : (i_batch + 1) * batch_size_int]
                for pts in int_col_params
            ]
            int_col_weights_batch = [
                weights[
                    i_batch * batch_size_int : (i_batch + 1) * batch_size_int
                ]
                for weights in int_col_weights
            ]
            bound_col_pts_batch = [
                pts[
                    i_batch
                    * batch_size_bound : (i_batch + 1)
                    * batch_size_bound
                ]
                for pts in bound_col_pts
            ]
            bound_col_params_batch = [
                pts[
                    i_batch
                    * batch_size_bound : (i_batch + 1)
                    * batch_size_bound
                ]
                for pts in bound_col_params
            ]
            bound_col_weights_batch = [
                weights[
                    i_batch
                    * batch_size_bound : (i_batch + 1)
                    * batch_size_bound
                ]
                for weights in bound_col_weights
            ]
            x_trainList_batch = [
                x[
                    i_batch
                    * batch_size_data : (i_batch + 1)
                    * batch_size_data,
                    :,
                ]
                for x in x_trainList[: self.ind_cs_offset_data]
            ]
            x_cs_trainList_batch = [
                x[
                    i_batch
                    * batch_size_data : (i_batch + 1)
                    * batch_size_data,
                    :,
                ]
                for x in x_trainList[self.ind_cs_offset_data :]
            ]
            x_params_trainList_batch = [
                x[
                    i_batch
                    * batch_size_data : (i_batch + 1)
                    * batch_size_data,
                    :,
                ]
                for x in x_params_trainList
            ]
            y_trainList_batch = [
                y[
                    i_batch
                    * batch_size_data : (i_batch + 1)
                    * batch_size_data,
                    :,
                ]
                for y in y_trainList
            ]
            data_col_weights_batch = [
                weights[
                    i_batch * batch_size_data : (i_batch + 1) * batch_size_data
                ]
                for weights in data_col_weights
            ]
            reg_col_pts_batch = [
                pts[i_batch * batch_size_reg : (i_batch + 1) * batch_size_reg]
                for pts in reg_col_pts
            ]
            reg_col_weights_batch = [
                weights[
                    i_batch * batch_size_reg : (i_batch + 1) * batch_size_reg
                ]
                for weights in reg_col_weights
            ]
            with tf.GradientTape() as tape:
                self.set_weights(w, self.sizes_w, self.sizes_b)
                interiorTerms = self.interior_loss(
                    int_col_pts_batch, int_col_params_batch, tmax
                )
                boundaryTerms = self.boundary_loss(
                    bound_col_pts_batch, bound_col_params_batch, tmax
                )
                dataTerms = self.data_loss(
                    x_trainList_batch,
                    x_cs_trainList_batch,
                    x_params_trainList_batch,
                    y_trainList_batch,
                )
                regularizationTerms = self.regularization_loss(
                    reg_col_pts_batch, tmax
                )
                interiorTerms_rescaled = [
                    interiorTerm[0] * resc
                    for (interiorTerm, resc) in zip(
                        interiorTerms, self.interiorTerms_rescale
                    )
                ]
                boundaryTerms_rescaled = [
                    boundaryTerm[0] * resc
                    for (boundaryTerm, resc) in zip(
                        boundaryTerms, self.boundaryTerms_rescale
                    )
                ]
                dataTerms_rescaled = [
                    dataTerm[0] * resc
                    for (dataTerm, resc) in zip(
                        dataTerms, self.dataTerms_rescale
                    )
                ]
                regularizationTerms_rescaled = [
                    regularizationTerm[0] * resc
                    for (regularizationTerm, resc) in zip(
                        regularizationTerms, self.regTerms_rescale
                    )
                ]
                (
                    loss_value,
                    int_loss,
                    bound_loss,
                    data_loss,
                    reg_loss,
                ) = loss_fn_lbfgs_SA(
                    interiorTerms_rescaled,
                    boundaryTerms_rescaled,
                    dataTerms_rescaled,
                    regularizationTerms_rescaled,
                    int_col_weights_batch,
                    bound_col_weights_batch,
                    data_col_weights_batch,
                    reg_col_weights_batch,
                    self.alpha,
                )
            (
                loss_value_unweighted,
                int_loss_unweighted,
                bound_loss_unweighted,
                data_loss_unweighted,
                reg_loss_unweighted,
            ) = loss_fn_lbfgs(
                interiorTerms_rescaled,
                boundaryTerms_rescaled,
                dataTerms_rescaled,
                regularizationTerms_rescaled,
                self.alpha,
            )
            grad = tape.gradient(loss_value, self.model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            accumulatedGradient += grad_flat / n_batch
            accumulatedLoss += loss_value / n_batch
            accumulatedUnweightedLoss += loss_value_unweighted / n_batch
            accumulatedLossInt += int_loss / n_batch
            accumulatedLossBound += bound_loss / n_batch
            accumulatedLossData += data_loss / n_batch
            accumulatedLossReg += reg_loss / n_batch

        return (
            accumulatedLoss,
            accumulatedGradient,
            accumulatedUnweightedLoss,
            interiorTerms_rescaled,
            boundaryTerms_rescaled,
            dataTerms_rescaled,
            regularizationTerms_rescaled,
            accumulatedLossInt,
            accumulatedLossBound,
            accumulatedLossData,
            accumulatedLossReg,
        )

    return loss_and_flat_grad


# L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
def get_loss_and_flat_grad(
    self,
    int_col_pts,
    int_col_params,
    bound_col_pts,
    bound_col_params,
    reg_col_pts,
    reg_col_params,
    x_trainList,
    x_params_trainList,
    y_trainList,
    n_batch=1,
    tmax=None,
):
    def loss_and_flat_grad(w):
        accumulatedGradient = 0
        accumulatedLoss = 0
        accumulatedLossInt = 0
        accumulatedLossBound = 0
        accumulatedLossData = 0
        accumulatedLossReg = 0
        batch_size_int = self.batch_size_int_lbfgs
        batch_size_bound = self.batch_size_bound_lbfgs
        batch_size_data = self.batch_size_data_lbfgs
        batch_size_reg = self.batch_size_reg_lbfgs
        for i_batch in range(n_batch):
            int_col_pts_batch = [
                pts[i_batch * batch_size_int : (i_batch + 1) * batch_size_int]
                for pts in int_col_pts
            ]
            int_col_params_batch = [
                pts[i_batch * batch_size_int : (i_batch + 1) * batch_size_int]
                for pts in int_col_params
            ]
            bound_col_pts_batch = [
                pts[
                    i_batch
                    * batch_size_bound : (i_batch + 1)
                    * batch_size_bound
                ]
                for pts in bound_col_pts
            ]
            bound_col_params_batch = [
                pts[
                    i_batch
                    * batch_size_bound : (i_batch + 1)
                    * batch_size_bound
                ]
                for pts in bound_col_params
            ]
            x_trainList_batch = [
                x[
                    i_batch
                    * batch_size_data : (i_batch + 1)
                    * batch_size_data,
                    :,
                ]
                for x in x_trainList[: self.ind_cs_offset_data]
            ]
            x_cs_trainList_batch = [
                x[
                    i_batch
                    * batch_size_data : (i_batch + 1)
                    * batch_size_data,
                    :,
                ]
                for x in x_trainList[self.ind_cs_offset_data :]
            ]
            x_params_trainList_batch = [
                x[
                    i_batch
                    * batch_size_data : (i_batch + 1)
                    * batch_size_data,
                    :,
                ]
                for x in x_params_trainList
            ]
            y_trainList_batch = [
                y[
                    i_batch
                    * batch_size_data : (i_batch + 1)
                    * batch_size_data,
                    :,
                ]
                for y in y_trainList
            ]
            reg_col_pts_batch = [
                pts[i_batch * batch_size_reg : (i_batch + 1) * batch_size_reg]
                for pts in reg_col_pts
            ]
            with tf.GradientTape() as tape:
                self.set_weights(w, self.sizes_w, self.sizes_b)
                interiorTerms = self.interior_loss(
                    int_col_pts_batch, int_col_params_batch, tmax
                )
                boundaryTerms = self.boundary_loss(
                    bound_col_pts_batch, bound_col_params_batch, tmax
                )
                dataTerms = self.data_loss(
                    x_trainList_batch,
                    x_cs_trainList_batch,
                    x_params_trainList_batch,
                    y_trainList_batch,
                )
                regularizationTerms = self.regularization_loss(
                    reg_col_pts_batch, tmax
                )
                interiorTerms_rescaled = [
                    interiorTerm[0] * resc
                    for (interiorTerm, resc) in zip(
                        interiorTerms, self.interiorTerms_rescale
                    )
                ]
                boundaryTerms_rescaled = [
                    boundaryTerm[0] * resc
                    for (boundaryTerm, resc) in zip(
                        boundaryTerms, self.boundaryTerms_rescale
                    )
                ]
                dataTerms_rescaled = [
                    dataTerm[0] * resc
                    for (dataTerm, resc) in zip(
                        dataTerms, self.dataTerms_rescale
                    )
                ]
                regularizationTerms_rescaled = [
                    regularizationTerm[0] * resc
                    for (regularizationTerm, resc) in zip(
                        regularizationTerms, self.regTerms_rescale
                    )
                ]
                (
                    loss_value,
                    int_loss,
                    bound_loss,
                    data_loss,
                    reg_loss,
                ) = loss_fn_lbfgs(
                    interiorTerms_rescaled,
                    boundaryTerms_rescaled,
                    dataTerms_rescaled,
                    regularizationTerms_rescaled,
                    self.alpha,
                )
            grad = tape.gradient(loss_value, self.model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            accumulatedGradient += grad_flat / n_batch
            accumulatedLoss += loss_value / n_batch
            accumulatedLossInt += int_loss / n_batch
            accumulatedLossBound += bound_loss / n_batch
            accumulatedLossData += data_loss / n_batch
            accumulatedLossReg += reg_loss / n_batch
        return (
            accumulatedLoss,
            accumulatedGradient,
            interiorTerms_rescaled,
            boundaryTerms_rescaled,
            dataTerms_rescaled,
            regularizationTerms_rescaled,
            accumulatedLossInt,
            accumulatedLossBound,
            accumulatedLossData,
            accumulatedLossReg,
        )

    return loss_and_flat_grad


def setResidualRescaling(self):
    # useful variables
    ce = self.params["ce0"]
    cs = np.float64(1.0 / 2.0) * (self.params["cs_a0"] + self.params["cs_c0"])
    cs_a = self.params["cs_a0"]
    cs_c = self.params["cs_c0"]
    Ds_a = np.float64(self.params["D_s_a"](self.params["T"], self.params["R"]))
    Ds_c = np.float64(
        self.params["D_s_c"](
            cs_c,
            self.params["T"],
            self.params["R"],
            self.params["cscamax"],
            np.float64(1.0),
        )
    )
    F = self.params["F"]
    R = self.params["rescale_R"]
    R_a = self.params["Rs_a"]
    R_c = self.params["Rs_c"]

    I = np.abs(self.params["I_discharge"])
    A_a = self.params["A_a"]
    A_c = self.params["A_c"]
    j_a = abs(self.params["j_a"])
    j_c = abs(self.params["j_c"])

    # Interior Residuals
    self.phie_transp_resc = np.float64(1.0) / j_a
    self.phis_c_transp_resc = np.float64(1.0) / j_c
    self.cs_a_transp_resc = np.float64(1.0) / (Ds_a * cs_a)
    self.cs_c_transp_resc = np.float64(1.0) / (Ds_c * cs_c)

    # Interior points
    if self.activeInt:
        self.interiorTerms_rescale = [
            np.float64(1.0) * abs(self.phie_transp_resc),
            np.float64(1.0) * abs(self.phis_c_transp_resc),
            np.float64(10.0) * abs(self.cs_a_transp_resc),
            np.float64(20.0) * abs(self.cs_c_transp_resc),
        ]
    else:
        self.interiorTerms_rescale = [np.float64(0.0)]

    # Boundary Residuals
    self.cs_a_bound_resc = R_a / cs_a
    self.cs_c_bound_resc = R_c / cs_c
    self.cs_a_bound_j_resc = Ds_a / j_a
    self.cs_c_bound_j_resc = Ds_c / j_c
    if self.activeBound:
        self.boundaryTerms_rescale = [
            np.float64(1.0) * abs(self.cs_a_bound_resc),
            np.float64(1.0) * abs(self.cs_c_bound_resc),
            np.float64(2.5) * abs(self.cs_a_bound_j_resc),
            np.float64(5.0) * abs(self.cs_c_bound_j_resc),
        ]
    else:
        self.boundaryTerms_rescale = [np.float64(0.0)]

    # Data Residuals
    self.n_data_terms = 4
    if self.activeData:
        self.dataTerms_rescale = [0 for _ in range(self.n_data_terms)]
        self.dataTerms_rescale[self.ind_phie_data] = abs(
            np.float64(1.0 / self.params["rescale_phie"])
        )
        self.dataTerms_rescale[self.ind_phis_c_data] = abs(
            np.float64(5.0 / self.params["rescale_phis_c"])
        )
        self.dataTerms_rescale[self.ind_cs_a_data] = abs(
            np.float64(1.0 / self.params["rescale_cs_a"])
        )
        self.dataTerms_rescale[self.ind_cs_c_data] = abs(
            np.float64(5.0 / self.params["rescale_cs_c"])
        )
        self.csDataTerms_ind = [self.ind_cs_a_data, self.ind_cs_c_data]
    else:
        self.dataTerms_rescale = [np.float64(0.0)]
        self.csDataTerms_ind = []

    # Regularization Residuals
    if self.activeReg:
        self.regTerms_rescale = [np.float64(0.0)]
    else:
        self.regTerms_rescale = [np.float64(0.0)]

    np.savez(
        "rescaleTF_int",
        phie=np.float64(1 / abs(self.phie_transp_resc)),
        phis_c=np.float64(1 / abs(self.phis_c_transp_resc)),
        cs_a=np.float64(1 / abs(self.cs_a_transp_resc)),
        cs_c=np.float64(1 / abs(self.cs_c_transp_resc)),
        cs_a_halfway=np.float64(1 / abs(self.cs_a_transp_resc)),
        cs_c_halfway=np.float64(1 / abs(self.cs_c_transp_resc)),
    )

    np.savez(
        "rescaleTF_bound",
        cs_a_r0=np.float64(1 / abs(self.cs_a_bound_resc)),
        cs_c_r0=np.float64(1 / abs(self.cs_c_bound_resc)),
        cs_a_rRs=np.float64(1 / abs(self.cs_a_bound_j_resc)),
        cs_c_rRs=np.float64(1 / abs(self.cs_c_bound_j_resc)),
    )

    return


@conditional_decorator(tf.function, optimized)
def data_loss(
    self,
    x_batch_trainList,
    x_cs_batch_trainList,
    x_params_batch_trainList,
    y_batch_trainList,
):

    if not self.activeData:
        return [[np.float64(0.0)]]

    # rescale
    resc_t = self.params["rescale_T"]
    resc_r = self.params["rescale_R"]

    dummyR = tf.zeros(
        x_batch_trainList[self.ind_phie_data][:, self.ind_t].shape,
        dtype=tf.dtypes.float64,
    )
    phie_pred_non_rescaled = self.model(
        [
            x_batch_trainList[self.ind_phie_data][:, self.ind_t] / resc_t,
            dummyR,
            x_params_batch_trainList[self.ind_phie_data][:, self.ind_deg_i0_a]
            / self.resc_params[self.ind_deg_i0_a],
            x_params_batch_trainList[self.ind_phie_data][:, self.ind_deg_ds_c]
            / self.resc_params[self.ind_deg_ds_c],
        ],
        training=True,
    )[self.ind_phie]

    shape_i0_a = x_batch_trainList[self.ind_phie_data][:, self.ind_t].shape
    i0_a_phie = self.params["i0_a"](
        None,
        self.params["ce0"] * tf.ones(shape_i0_a, dtype=tf.dtypes.float64),
        None,
        None,
        None,
        None,
        x_params_batch_trainList[self.ind_phie_data][:, self.ind_deg_i0_a],
    )

    phie_pred_rescaled = self.rescalePhie(
        phie_pred_non_rescaled,
        x_batch_trainList[self.ind_phie_data][:, self.ind_t],
        i0_a_phie,
    )

    dummyR = tf.zeros(
        x_batch_trainList[self.ind_phis_c_data][:, self.ind_t].shape,
        dtype=tf.dtypes.float64,
    )
    phis_c_pred_non_rescaled = self.model(
        [
            x_batch_trainList[self.ind_phis_c_data][:, self.ind_t] / resc_t,
            dummyR,
            x_params_batch_trainList[self.ind_phis_c_data][
                :, self.ind_deg_i0_a
            ]
            / self.resc_params[self.ind_deg_i0_a],
            x_params_batch_trainList[self.ind_phis_c_data][
                :, self.ind_deg_ds_c
            ]
            / self.resc_params[self.ind_deg_ds_c],
        ],
        training=True,
    )[self.ind_phis_c]
    shape_i0_a = x_batch_trainList[self.ind_phis_c_data][:, self.ind_t].shape
    i0_a_phis = self.params["i0_a"](
        None,
        self.params["ce0"] * tf.ones(shape_i0_a, dtype=tf.dtypes.float64),
        None,
        None,
        None,
        None,
        x_params_batch_trainList[self.ind_phis_c_data][:, self.ind_deg_i0_a],
    )
    phis_c_pred_rescaled = self.rescalePhis_c(
        phis_c_pred_non_rescaled,
        x_batch_trainList[self.ind_phis_c_data][:, self.ind_t],
        i0_a_phis,
    )

    cs_a_pred_non_rescaled = self.model(
        [
            x_cs_batch_trainList[self.ind_cs_a_data - self.ind_cs_offset_data][
                :, self.ind_t
            ]
            / resc_t,
            x_cs_batch_trainList[self.ind_cs_a_data - self.ind_cs_offset_data][
                :, self.ind_r
            ]
            / resc_r,
            x_params_batch_trainList[self.ind_cs_a_data][:, self.ind_deg_i0_a]
            / self.resc_params[self.ind_deg_i0_a],
            x_params_batch_trainList[self.ind_cs_a_data][:, self.ind_deg_ds_c]
            / self.resc_params[self.ind_deg_ds_c],
        ],
        training=True,
    )[self.ind_cs_a]
    cs_a_pred_rescaled = self.rescaleCs_a(
        cs_a_pred_non_rescaled,
        x_cs_batch_trainList[self.ind_cs_a_data - self.ind_cs_offset_data][
            :, self.ind_t
        ],
    )
    cs_c_pred_non_rescaled = self.model(
        [
            x_cs_batch_trainList[self.ind_cs_c_data - self.ind_cs_offset_data][
                :, self.ind_t
            ]
            / resc_t,
            x_cs_batch_trainList[self.ind_cs_c_data - self.ind_cs_offset_data][
                :, self.ind_r
            ]
            / resc_r,
            x_params_batch_trainList[self.ind_cs_c_data][:, self.ind_deg_i0_a]
            / self.resc_params[self.ind_deg_i0_a],
            x_params_batch_trainList[self.ind_cs_c_data][:, self.ind_deg_ds_c]
            / self.resc_params[self.ind_deg_ds_c],
        ],
        training=True,
    )[self.ind_cs_c]
    cs_c_pred_rescaled = self.rescaleCs_c(
        cs_c_pred_non_rescaled,
        x_cs_batch_trainList[self.ind_cs_c_data - self.ind_cs_offset_data][
            :, self.ind_t
        ],
    )

    return [
        [phie_pred_rescaled - y_batch_trainList[self.ind_phie_data]],
        [phis_c_pred_rescaled - y_batch_trainList[self.ind_phis_c_data]],
        [cs_a_pred_rescaled - y_batch_trainList[self.ind_cs_a_data]],
        [cs_c_pred_rescaled - y_batch_trainList[self.ind_cs_c_data]],
    ]


@conditional_decorator(tf.function, optimized)
def interior_loss(self, int_col_pts=None, int_col_params=None, tmax=None):

    if not self.activeInt:
        return [[np.float64(0.0)]]

    tmin_int = tf.math.minimum(self.tmin_int_bound, self.tmax)

    if self.collocationMode == "random":

        if self.gradualTime:
            t = tf.random.uniform(
                (self.batch_size_int, 1),
                minval=tmin_int,
                maxval=tmax,
                dtype=tf.dtypes.float64,
            )
        else:
            t = tf.random.uniform(
                (self.batch_size_int, 1),
                minval=tmin_int,
                maxval=self.tmax,
                dtype=tf.dtypes.float64,
            )

        r_a = tf.random.uniform(
            (self.batch_size_int, 1),
            minval=self.rmin,
            maxval=self.rmax_a,
            dtype=tf.dtypes.float64,
        )
        r_c = tf.random.uniform(
            (self.batch_size_int, 1),
            minval=self.rmin,
            maxval=self.rmax_c,
            dtype=tf.dtypes.float64,
        )
        dummyR = tf.zeros((self.batch_size_int, 1), dtype=tf.dtypes.float64)
        rSurf_a = self.rmax_a * tf.ones(
            (self.batch_size_int, 1), dtype=tf.dtypes.float64
        )
        rSurf_c = self.rmax_c * tf.ones(
            (self.batch_size_int, 1), dtype=tf.dtypes.float64
        )
        deg_i0_a = tf.random.uniform(
            (self.batch_size_int, 1),
            minval=self.params["deg_i0_a_min_eff"],
            maxval=self.params["deg_i0_a_max_eff"],
            dtype=tf.dtypes.float64,
        )
        deg_ds_c = tf.random.uniform(
            (self.batch_size_int, 1),
            minval=self.params["deg_ds_c_min_eff"],
            maxval=self.params["deg_ds_c_max_eff"],
            dtype=tf.dtypes.float64,
        )

    elif self.collocationMode == "fixed":

        if self.gradualTime:
            t = self.stretchT(
                int_col_pts[self.ind_int_col_t],
                tmin_int,
                self.firstTime,
                tmin_int,
                tmax,
            )
        else:
            t = int_col_pts[self.ind_int_col_t]

        r_a = int_col_pts[self.ind_int_col_r_a]
        rSurf_a = int_col_pts[self.ind_int_col_r_maxa]
        r_c = int_col_pts[self.ind_int_col_r_c]
        rSurf_c = int_col_pts[self.ind_int_col_r_maxc]
        dummyR = tf.zeros(r_a.shape, dtype=tf.dtypes.float64)
        dummy_par = tf.random.uniform(
            (self.batch_size_int, 1),
            minval=np.float64(1),
            maxval=np.float64(1),
            dtype=tf.dtypes.float64,
        )
        deg_i0_a = int_col_params[self.ind_int_col_params_deg_i0_a]
        deg_ds_c = int_col_params[self.ind_int_col_params_deg_ds_c]

    # rescale
    resc_t = self.params["rescale_T"]
    resc_r = self.params["rescale_R"]
    # constants
    ce = self.params["ce0"] * tf.ones(t.shape, dtype=tf.dtypes.float64)
    phis_a = tf.zeros(t.shape, dtype=tf.dtypes.float64)

    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True
    ) as tape:
        # Watch some tensors
        tape.watch(r_a)
        tape.watch(r_c)
        tape.watch(t)

        # Feed forward
        output_a = self.model(
            [
                t / resc_t,
                r_a / resc_r,
                deg_i0_a / self.resc_params[self.ind_deg_i0_a],
                deg_ds_c / self.resc_params[self.ind_deg_ds_c],
            ],
            training=True,
        )
        output_c = self.model(
            [
                t / resc_t,
                r_c / resc_r,
                deg_i0_a / self.resc_params[self.ind_deg_i0_a],
                deg_ds_c / self.resc_params[self.ind_deg_ds_c],
            ],
            training=True,
        )
        output_surf_a = self.model(
            [
                t / resc_t,
                rSurf_a / resc_r,
                deg_i0_a / self.resc_params[self.ind_deg_i0_a],
                deg_ds_c / self.resc_params[self.ind_deg_ds_c],
            ],
            training=True,
        )
        output_surf_c = self.model(
            [
                t / resc_t,
                rSurf_c / resc_r,
                deg_i0_a / self.resc_params[self.ind_deg_i0_a],
                deg_ds_c / self.resc_params[self.ind_deg_ds_c],
            ],
            training=True,
        )

        cse_a = self.rescaleCs_a(output_surf_a[self.ind_cs_a], t)

        i0_a = self.params["i0_a"](
            cse_a,
            ce,
            self.params["T"],
            self.params["alpha_a"],
            self.params["csanmax"],
            self.params["R"],
            deg_i0_a,
        )
        phie = self.rescalePhie(output_a[self.ind_phie], t, i0_a)
        phis_c = self.rescalePhis_c(output_c[self.ind_phis_c], t, i0_a)

        cs_a = self.rescaleCs_a(output_a[self.ind_cs_a], t)

        cs_c = self.rescaleCs_c(output_c[self.ind_cs_c], t)
        cse_c = self.rescaleCs_c(output_surf_c[self.ind_cs_c], t)

        # Equations at anode
        # ~~~~ j
        eta_a = (
            phis_a
            - phie
            - self.params["Uocp_a"](cse_a, self.params["csanmax"])
        )
        if not self.linearizeJ:
            exp1_a = tf.exp(
                (np.float64(1.0) - self.params["alpha_a"])
                * self.params["F"]
                * eta_a
                / (self.params["R"] * self.params["T"])
            )
            exp2_a = tf.exp(
                -self.params["alpha_a"]
                * self.params["F"]
                * eta_a
                / (self.params["R"] * self.params["T"])
            )
            j_a = (i0_a / self.params["F"]) * (exp1_a - exp2_a)
        else:
            j_a = i0_a * eta_a / (self.params["R"] * self.params["T"])

        j_a_rhs = self.params["j_a"]

        # ~~~~ cs
        cs_a_r = tape.gradient(cs_a, r_a)
        ds_a = self.params["D_s_a"](self.params["T"], self.params["R"])
        # cs_a_r_Dsr2_r =  tape.gradient(cs_a_r_Dsr2, r_a)
        # cs_a_r_Dsr2_r_r2 = cs_a_r_Dsr2_r / (r_a **2)

        # Equations at cathode
        # ~~~~ j
        i0_c = self.params["i0_c"](
            cse_c,
            ce,
            self.params["T"],
            self.params["alpha_c"],
            self.params["cscamax"],
            self.params["R"],
        )
        eta_c = (
            phis_c
            - phie
            - self.params["Uocp_c"](cse_c, self.params["cscamax"])
        )
        if not self.linearizeJ:
            exp1_c = tf.exp(
                (np.float64(1.0) - self.params["alpha_c"])
                * self.params["F"]
                * eta_c
                / (self.params["R"] * self.params["T"])
            )
            exp2_c = tf.exp(
                -self.params["alpha_c"]
                * self.params["F"]
                * eta_c
                / (self.params["R"] * self.params["T"])
            )
            j_c = (i0_c / self.params["F"]) * (exp1_c - exp2_c)
        else:
            j_c = i0_c * eta_c / (self.params["R"] * self.params["T"])

        j_c_rhs = self.params["j_c"]

        # ~~~~ cs
        cs_c_r = tape.gradient(cs_c, r_c)
        ds_c = self.params["D_s_c"](
            cs_c,
            self.params["T"],
            self.params["R"],
            self.params["cscamax"],
            deg_ds_c,
        )
        # cs_c_r_Dsr2_r = tape.gradient(cs_c_r_Dsr2, r_c)
        # cs_c_r_Dsr2_r_r2 = cs_c_r_Dsr2_r / (r_c **2)

    # Equations at anode
    # ~~~~ cs
    cs_a_t = tape.gradient(cs_a, t)
    cs_a_r_r = tape.gradient(cs_a_r, r_a)
    # ds_a_r = tape.gradient(ds_a, r_a)

    # Equations at cathode
    # ~~~~ cs
    cs_c_t = tape.gradient(cs_c, t)
    cs_c_r_r = tape.gradient(cs_c_r, r_c)
    # ds_c_r = tape.gradient(ds_c, r_c)

    # Letting the tape go
    del tape

    # List of residuals
    return [
        [j_a - j_a_rhs],
        [j_c - j_c_rhs],
        [
            r_a * r_a * cs_a_t
            - r_a * r_a * cs_a_r_r * ds_a
            - r_a * np.float64(2.0) * ds_a * cs_a_r
        ],  # cs at anode
        [
            r_c * r_c * cs_c_t / deg_ds_c
            - r_c * r_c * cs_c_r_r * ds_c / deg_ds_c
            - r_c * np.float64(2.0) * ds_c * cs_c_r / deg_ds_c
        ],  # cs at cathode
        # [r_a * cs_a_t - r_a * cs_a_r_r * ds_a - np.float64(2.0) * ds_a * cs_a_r - r_a * ds_a_r * cs_a_r],  # cs at anode
        # [r_c * cs_c_t - r_c * cs_c_r_r * ds_c - np.float64(2.0) * ds_c * cs_c_r - r_c * ds_c_r * cs_c_r],  # cs at cathode
    ]


@conditional_decorator(tf.function, optimized)
def boundary_loss(self, bound_col_pts=None, bound_col_params=None, tmax=None):

    if not self.activeBound:
        return [[np.float64(0.0)]]

    tmin_bound = tf.math.minimum(self.tmin_int_bound, self.tmax)

    if self.collocationMode == "random":

        if self.gradualTime:
            t_bound = tf.random.uniform(
                (self.batch_size_bound, 1),
                minval=tmin_bound,
                maxval=tmax,
                dtype=tf.dtypes.float64,
            )
        else:
            t_bound = tf.random.uniform(
                (self.batch_size_bound, 1),
                minval=tmin_bound,
                maxval=self.tmax,
                dtype=tf.dtypes.float64,
            )

        r_0_bound = tf.zeros(
            (self.batch_size_bound, 1), dtype=tf.dtypes.float64
        )
        r_max_a_bound = self.rmax_a * tf.ones(
            (self.batch_size_bound, 1), dtype=tf.dtypes.float64
        )
        r_max_c_bound = self.rmax_c * tf.ones(
            (self.batch_size_bound, 1), dtype=tf.dtypes.float64
        )
        deg_i0_rmax_bound = tf.random.uniform(
            (self.batch_size_bound, 1),
            minval=self.params["deg_i0_a_min_eff"],
            maxval=self.params["deg_i0_a_max_eff"],
            dtype=tf.dtypes.float64,
        )
        deg_ds_rmax_c_bound = tf.random.uniform(
            (self.batch_size_bound, 1),
            minval=self.params["deg_ds_c_min_eff"],
            maxval=self.params["deg_ds_c_max_eff"],
            dtype=tf.dtypes.float64,
        )

    if self.collocationMode == "fixed":

        if self.gradualTime:
            t_bound = self.stretchT(
                bound_col_pts[self.ind_bound_col_t],
                tmin_bound,
                self.firstTime,
                tmin_bound,
                tmax,
            )
        else:
            t_bound = bound_col_pts[self.ind_bound_col_t]

        r_0_bound = bound_col_pts[self.ind_bound_col_r_min]
        r_max_a_bound = bound_col_pts[self.ind_bound_col_r_maxa]
        r_max_c_bound = bound_col_pts[self.ind_bound_col_r_maxc]
        deg_i0_rmax_bound = bound_col_params[
            self.ind_bound_col_params_deg_i0_a
        ]
        deg_ds_rmax_c_bound = bound_col_params[
            self.ind_bound_col_params_deg_ds_c
        ]

    # rescale
    resc_t = self.params["rescale_T"]
    resc_r = self.params["rescale_R"]

    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True
    ) as tape:

        tape.watch(r_0_bound)
        tape.watch(r_max_a_bound)
        tape.watch(r_max_c_bound)

        # Feed forward
        output_r0_a_bound = self.model(
            [
                t_bound / resc_t,
                r_0_bound / resc_r,
                deg_i0_rmax_bound / self.resc_params[self.ind_deg_i0_a],
                deg_ds_rmax_c_bound / self.resc_params[self.ind_deg_ds_c],
            ],
            training=True,
        )
        output_r0_c_bound = self.model(
            [
                t_bound / resc_t,
                r_0_bound / resc_r,
                deg_i0_rmax_bound / self.resc_params[self.ind_deg_i0_a],
                deg_ds_rmax_c_bound / self.resc_params[self.ind_deg_ds_c],
            ],
            training=True,
        )
        output_rmax_a_bound = self.model(
            [
                t_bound / resc_t,
                r_max_a_bound / resc_r,
                deg_i0_rmax_bound / self.resc_params[self.ind_deg_i0_a],
                deg_ds_rmax_c_bound / self.resc_params[self.ind_deg_ds_c],
            ],
            training=True,
        )
        output_rmax_c_bound = self.model(
            [
                t_bound / resc_t,
                r_max_c_bound / resc_r,
                deg_i0_rmax_bound / self.resc_params[self.ind_deg_i0_a],
                deg_ds_rmax_c_bound / self.resc_params[self.ind_deg_ds_c],
            ],
            training=True,
        )

        # Output
        cs_r0_a_bound = self.rescaleCs_a(
            output_r0_a_bound[self.ind_cs_a], t_bound
        )
        cs_r0_c_bound = self.rescaleCs_c(
            output_r0_c_bound[self.ind_cs_c], t_bound
        )
        cs_rmax_a_bound = self.rescaleCs_a(
            output_rmax_a_bound[self.ind_cs_a], t_bound
        )
        cs_rmax_c_bound = self.rescaleCs_c(
            output_rmax_c_bound[self.ind_cs_c], t_bound
        )

        ds_rmax_a_bound = self.params["D_s_a"](
            self.params["T"], self.params["R"]
        )

        ds_rmax_c_bound = self.params["D_s_c"](
            cs_rmax_c_bound,
            self.params["T"],
            self.params["R"],
            self.params["cscamax"],
            deg_ds_rmax_c_bound,
        )

        j_a = self.params["j_a"]
        j_c = self.params["j_c"]

    # boundary Eq

    cs_r0_a_bound_r = tape.gradient(cs_r0_a_bound, r_0_bound)
    cs_r0_c_bound_r = tape.gradient(cs_r0_c_bound, r_0_bound)
    cs_rmax_a_bound_r = tape.gradient(cs_rmax_a_bound, r_max_a_bound)
    cs_rmax_c_bound_r = tape.gradient(cs_rmax_c_bound, r_max_c_bound)

    # Letting the tape go
    del tape

    # List of relevant output
    return [
        [cs_r0_a_bound_r],
        [cs_r0_c_bound_r],
        [cs_rmax_a_bound_r + j_a / ds_rmax_a_bound],
        [
            deg_ds_rmax_c_bound * cs_rmax_c_bound_r
            + deg_ds_rmax_c_bound * j_c / ds_rmax_c_bound
        ],
    ]


@conditional_decorator(tf.function, optimized)
def regularization_loss(self, reg_col_pts=None, tmax=None):

    if not self.activeReg:
        return [[np.float64(0.0)]]

    return [[np.float64(0.0)]]
