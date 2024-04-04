import argument
import keras
import numpy as np
import tensorflow as tf
from conditionalDecorator import conditional_decorator

keras.backend.set_floatx("float64")

# Read command line arguments
args = argument.initArg()


# @conditional_decorator(tf.function,optimized)
def rescalePhie(self, phie, t, deg_i0_a, deg_ds_c):
    # t and deg inputs have physical units
    # State variables inputs are rescaled
    resc_phie = self.params["rescale_phie"]
    t_reshape = tf.reshape(t, tf.shape(phie))
    deg_i0_a_reshape = tf.reshape(deg_i0_a, tf.shape(phie))
    deg_ds_c_reshape = tf.reshape(deg_ds_c, tf.shape(phie))

    if self.use_hnntime:
        phie_start = self.get_phie_hnntime(deg_i0_a_reshape, deg_ds_c_reshape)
        timeDistance = np.float64(1.0) - tf.exp(
            -(t_reshape - self.hnntime_val) / self.hard_IC_timescale
        )
    else:
        phie_start = self.get_phie0(deg_i0_a_reshape)
        timeDistance = np.float64(1.0) - tf.exp(
            -(t_reshape) / self.hard_IC_timescale
        )

    offset = np.float64(0.0)
    phie_nn = phie
    if self.use_hnn:
        phie_hnn = self.get_phie_hnn(
            t_reshape, deg_i0_a_reshape, deg_ds_c_reshape
        )
        offset = phie_hnn - phie_start
        resc_phie *= np.float64(0.1)

    return (resc_phie * phie_nn + offset) * timeDistance + phie_start


# @conditional_decorator(tf.function,optimized)
def rescalePhis_c(self, phis_c, t, deg_i0_a, deg_ds_c):
    # t and deg inputs have physical units
    # State variables inputs are rescaled
    resc_phis_c = self.params["rescale_phis_c"]
    t_reshape = tf.reshape(t, tf.shape(phis_c))
    deg_i0_a_reshape = tf.reshape(deg_i0_a, tf.shape(phis_c))
    deg_ds_c_reshape = tf.reshape(deg_ds_c, tf.shape(phis_c))

    if self.use_hnntime:
        phis_c_start = self.get_phis_c_hnntime(
            deg_i0_a_reshape, deg_ds_c_reshape
        )
        timeDistance = np.float64(1.0) - tf.exp(
            -(t_reshape - self.hnntime_val) / self.hard_IC_timescale
        )
    else:
        phis_c_start = self.get_phis_c0(deg_i0_a_reshape)
        timeDistance = np.float64(1.0) - tf.exp(
            -(t_reshape) / self.hard_IC_timescale
        )

    offset = np.float64(0.0)
    phis_c_nn = phis_c
    if self.use_hnn:
        phis_c_hnn = self.get_phis_c_hnn(
            t_reshape, deg_i0_a_reshape, deg_ds_c_reshape
        )
        offset = phis_c_hnn - phis_c_start
        resc_phis_c *= np.float64(0.1)

    return (resc_phis_c * phis_c_nn + offset) * timeDistance + phis_c_start


# @conditional_decorator(tf.function,optimized)
def rescaleCs_a(self, cs_a, t, r, deg_i0_a, deg_ds_c, clip=True):
    # t, r and deg inputs have physical units
    # State variables inputs are rescaled
    resc_cs_a = self.params["rescale_cs_a"]
    t_reshape = tf.reshape(t, tf.shape(cs_a))
    r_reshape = tf.reshape(r, tf.shape(cs_a))
    deg_i0_a_reshape = tf.reshape(deg_i0_a, tf.shape(cs_a))
    deg_ds_c_reshape = tf.reshape(deg_ds_c, tf.shape(cs_a))

    if self.use_hnntime:
        cs_a_start = self.get_cs_a_hnntime(
            r_reshape, deg_i0_a_reshape, deg_ds_c_reshape
        )
        timeDistance = np.float64(1.0) - tf.exp(
            -(t_reshape - self.hnntime_val) / self.hard_IC_timescale
        )
    else:
        cs_a_start = self.cs_a0
        timeDistance = np.float64(1.0) - tf.exp(
            -(t_reshape) / self.hard_IC_timescale
        )
    resc_cs_a = -cs_a_start

    offset = np.float64(0.0)
    if self.use_hnn:
        cs_a_hnn = self.get_cs_a_hnn(
            t_reshape, r_reshape, deg_i0_a_reshape, deg_ds_c_reshape
        )
        offset = cs_a_hnn - cs_a_start
        cs_a_nn = cs_a * np.float64(0.01)
    else:
        cs_a_nn = tf.math.sigmoid(cs_a)

    if clip:
        return tf.clip_by_value(
            (resc_cs_a * cs_a_nn + offset) * timeDistance + cs_a_start,
            np.float64(0.0),
            self.params["csanmax"],
        )
    else:
        return (resc_cs_a * cs_a_nn + offset) * timeDistance + cs_a_start


# @conditional_decorator(tf.function,optimized)
def rescaleCs_c(self, cs_c, t, r, deg_i0_a, deg_ds_c, clip=True):
    # t, r and deg inputs have physical units
    # State variables inputs are rescaled
    resc_cs_c = self.params["rescale_cs_c"]
    t_reshape = tf.reshape(t, tf.shape(cs_c))
    r_reshape = tf.reshape(r, tf.shape(cs_c))
    deg_i0_a_reshape = tf.reshape(deg_i0_a, tf.shape(cs_c))
    deg_ds_c_reshape = tf.reshape(deg_ds_c, tf.shape(cs_c))

    if self.use_hnntime:
        cs_c_start = self.get_cs_c_hnntime(
            r_reshape, deg_i0_a_reshape, deg_ds_c_reshape
        )
        timeDistance = np.float64(1.0) - tf.exp(
            -(t_reshape - self.hnntime_val) / self.hard_IC_timescale
        )
    else:
        cs_c_start = self.cs_c0
        timeDistance = np.float64(1.0) - tf.exp(
            -(t_reshape) / self.hard_IC_timescale
        )
    resc_cs_c = self.params["cscamax"] - cs_c_start

    offset = np.float64(0.0)
    if self.use_hnn:
        cs_c_hnn = self.get_cs_c_hnn(
            t_reshape, r_reshape, deg_i0_a_reshape, deg_ds_c_reshape
        )
        offset = cs_c_hnn - cs_c_start
        cs_c_nn = cs_c * np.float64(0.01)
    else:
        cs_c_nn = tf.math.sigmoid(cs_c)

    if clip:
        return tf.clip_by_value(
            (resc_cs_c * cs_c_nn + offset) * timeDistance + cs_c_start,
            np.float64(0.0),
            self.params["cscamax"],
        )
    else:
        return (resc_cs_c * cs_c_nn + offset) * timeDistance + cs_c_start


def get_phie0(self, deg_i0_a):
    i0_a = self.params["i0_a"](
        self.params["cs_a0"]
        * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
        self.params["ce0"]
        * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
        self.params["T"],
        self.params["alpha_a"],
        self.params["csanmax"],
        self.params["R"],
        deg_i0_a,
    )
    return self.params["phie0"](
        i0_a,
        self.params["j_a"],
        self.params["F"],
        self.params["R"],
        self.params["T"],
        self.params["Uocp_a0"],
    )


def get_phis_c0(self, deg_i0_a):
    i0_a = self.params["i0_a"](
        self.params["cs_a0"]
        * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
        self.params["ce0"]
        * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
        self.params["T"],
        self.params["alpha_a"],
        self.params["csanmax"],
        self.params["R"],
        deg_i0_a,
    )

    return self.params["phis_c0"](
        i0_a,
        self.params["j_a"],
        self.params["F"],
        self.params["R"],
        self.params["T"],
        self.params["Uocp_a0"],
        self.params["j_c"],
        self.params["i0_c0"],
        self.params["Uocp_c0"],
    )


# PHIE
def get_phie_hnn(self, t, deg_i0_a, deg_ds_c):
    if self.hnn_params is not None:
        return self.hnn.rescalePhie(
            self.hnn.model(
                [
                    t / self.hnn.params["rescale_T"],
                    tf.zeros(tf.shape(t), dtype=tf.dtypes.float64),
                    self.hnn.rescale_param(
                        self.fix_param(
                            deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a]
                        ),
                        self.hnn.ind_deg_i0_a,
                    ),
                    self.hnn.rescale_param(
                        self.fix_param(
                            deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c]
                        ),
                        self.hnn.ind_deg_ds_c,
                    ),
                ],
                training=False,
            )[self.hnn.ind_phie],
            t,
            self.fix_param(deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a]),
            self.fix_param(deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c]),
        )
    else:
        return self.hnn.rescalePhie(
            self.hnn.model(
                [
                    t / self.hnn.params["rescale_T"],
                    tf.zeros(tf.shape(t), dtype=tf.dtypes.float64),
                    self.hnn.rescale_param(deg_i0_a, self.hnn.ind_deg_i0_a),
                    self.hnn.rescale_param(deg_ds_c, self.hnn.ind_deg_ds_c),
                ],
                training=False,
            )[self.hnn.ind_phie],
            t,
            deg_i0_a,
            deg_ds_c,
        )


def get_phie_hnntime(self, deg_i0_a, deg_ds_c):
    return self.hnntime.rescalePhie(
        self.hnntime.model(
            [
                self.hnntime_val
                * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64)
                / self.hnntime.params["rescale_T"],
                tf.zeros(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
                self.hnntime.rescale_param(
                    deg_i0_a, self.hnntime.ind_deg_i0_a
                ),
                self.hnntime.rescale_param(
                    deg_ds_c, self.hnntime.ind_deg_ds_c
                ),
            ],
            training=False,
        )[self.hnntime.ind_phie],
        self.hnntime_val
        * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
        deg_i0_a,
        deg_ds_c,
    )


# PHISC
def get_phis_c_hnn(self, t, deg_i0_a, deg_ds_c):
    if self.hnn_params is not None:
        return self.hnn.rescalePhis_c(
            self.hnn.model(
                [
                    t / self.hnn.params["rescale_T"],
                    tf.zeros(tf.shape(t), dtype=tf.dtypes.float64),
                    self.hnn.rescale_param(
                        self.fix_param(
                            deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a]
                        ),
                        self.hnn.ind_deg_i0_a,
                    ),
                    self.hnn.rescale_param(
                        self.fix_param(
                            deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c]
                        ),
                        self.hnn.ind_deg_ds_c,
                    ),
                ],
                training=False,
            )[self.hnn.ind_phis_c],
            t,
            self.fix_param(deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a]),
            self.fix_param(deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c]),
        )
    else:
        return self.hnn.rescalePhis_c(
            self.hnn.model(
                [
                    t / self.hnn.params["rescale_T"],
                    tf.zeros(tf.shape(t), dtype=tf.dtypes.float64),
                    self.hnn.rescale_param(deg_i0_a, self.hnn.ind_deg_i0_a),
                    self.hnn.rescale_param(deg_ds_c, self.hnn.ind_deg_ds_c),
                ],
                training=False,
            )[self.hnn.ind_phis_c],
            t,
            deg_i0_a,
            deg_ds_c,
        )


def get_phis_c_hnntime(self, deg_i0_a, deg_ds_c):
    return self.hnntime.rescalePhis_c(
        self.hnntime.model(
            [
                self.hnntime_val
                * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64)
                / self.hnntime.params["rescale_T"],
                tf.zeros(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
                self.hnntime.rescale_param(
                    deg_i0_a, self.hnntime.ind_deg_i0_a
                ),
                self.hnntime.rescale_param(
                    deg_ds_c, self.hnntime.ind_deg_ds_c
                ),
            ],
            training=False,
        )[self.hnntime.ind_phis_c],
        self.hnntime_val
        * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
        deg_i0_a,
        deg_ds_c,
    )


# CSA
def get_cs_a_hnn(self, t, r, deg_i0_a, deg_ds_c):
    if self.hnn_params is not None:
        return self.hnn.rescaleCs_a(
            self.hnn.model(
                [
                    t / self.hnn.params["rescale_T"],
                    r / self.hnn.params["rescale_R"],
                    self.hnn.rescale_param(
                        self.fix_param(
                            deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a]
                        ),
                        self.hnn.ind_deg_i0_a,
                    ),
                    self.hnn.rescale_param(
                        self.fix_param(
                            deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c]
                        ),
                        self.hnn.ind_deg_ds_c,
                    ),
                ],
                training=False,
            )[self.hnn.ind_cs_a],
            t,
            r,
            self.fix_param(deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a]),
            self.fix_param(deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c]),
        )
    else:
        return self.hnn.rescaleCs_a(
            self.hnn.model(
                [
                    t / self.hnn.params["rescale_T"],
                    r / self.hnn.params["rescale_R"],
                    self.hnn.rescale_param(deg_i0_a, self.hnn.ind_deg_i0_a),
                    self.hnn.rescale_param(deg_ds_c, self.hnn.ind_deg_ds_c),
                ],
                training=False,
            )[self.hnn.ind_cs_a],
            t,
            r,
            deg_i0_a,
            deg_ds_c,
        )


def get_cs_a_hnntime(self, r, deg_i0_a, deg_ds_c):
    return self.hnntime.rescaleCs_a(
        self.hnntime.model(
            [
                self.hnntime_val
                * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64)
                / self.hnntime.params["rescale_T"],
                r / self.hnn.params["rescale_R"],
                self.hnntime.rescale_param(
                    deg_i0_a, self.hnntime.ind_deg_i0_a
                ),
                self.hnntime.rescale_param(
                    deg_ds_c, self.hnntime.ind_deg_ds_c
                ),
            ],
            training=False,
        )[self.hnntime.ind_cs_a],
        self.hnntime_val
        * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
        r / self.hnn.params["rescale_R"],
        deg_i0_a,
        deg_ds_c,
    )


# CSC
def get_cs_c_hnn(self, t, r, deg_i0_a, deg_ds_c):
    if self.hnn_params is not None:
        return self.hnn.rescaleCs_c(
            self.hnn.model(
                [
                    t / self.hnn.params["rescale_T"],
                    r / self.hnn.params["rescale_R"],
                    self.hnn.rescale_param(
                        self.fix_param(
                            deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a]
                        ),
                        self.hnn.ind_deg_i0_a,
                    ),
                    self.hnn.rescale_param(
                        self.fix_param(
                            deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c]
                        ),
                        self.hnn.ind_deg_ds_c,
                    ),
                ],
                training=False,
            )[self.hnn.ind_cs_c],
            t,
            r,
            self.fix_param(deg_i0_a, self.hnn_params[self.hnn.ind_deg_i0_a]),
            self.fix_param(deg_ds_c, self.hnn_params[self.hnn.ind_deg_ds_c]),
        )
    else:
        return self.hnn.rescaleCs_c(
            self.hnn.model(
                [
                    t / self.hnn.params["rescale_T"],
                    r / self.hnn.params["rescale_R"],
                    self.hnn.rescale_param(deg_i0_a, self.hnn.ind_deg_i0_a),
                    self.hnn.rescale_param(deg_ds_c, self.hnn.ind_deg_ds_c),
                ],
                training=False,
            )[self.hnn.ind_cs_c],
            t,
            r,
            deg_i0_a,
            deg_ds_c,
        )


def get_cs_c_hnntime(self, r, deg_i0_a, deg_ds_c):
    return self.hnntime.rescaleCs_c(
        self.hnntime.model(
            [
                self.hnntime_val
                * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64)
                / self.hnntime.params["rescale_T"],
                r / self.hnn.params["rescale_R"],
                self.hnntime.rescale_param(
                    deg_i0_a, self.hnntime.ind_deg_i0_a
                ),
                self.hnntime.rescale_param(
                    deg_ds_c, self.hnntime.ind_deg_ds_c
                ),
            ],
            training=False,
        )[self.hnntime.ind_cs_c],
        self.hnntime_val
        * tf.ones(tf.shape(deg_i0_a), dtype=tf.dtypes.float64),
        r / self.hnn.params["rescale_R"],
        deg_i0_a,
        deg_ds_c,
    )


def rescale_param(self, param, ind):
    return (param - self.params_min[ind]) / self.resc_params[ind]


def fix_param(self, param, param_val):
    return param_val * tf.ones(tf.shape(param), dtype=tf.dtypes.float64)


def unrescale_param(self, param_rescaled, ind):
    return param_rescaled * self.resc_params[ind] + self.params_min[ind]
