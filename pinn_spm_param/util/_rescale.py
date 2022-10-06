import argument
import numpy as np
import tensorflow as tf
from conditionalDecorator import conditional_decorator

tf.keras.backend.set_floatx("float64")

# Read command line arguments
args = argument.initArg()


# @conditional_decorator(tf.function,optimized)
def rescalePhie(self, phie, t, i0_a):
    # t and x inputs have physical units
    # State variables inputs are rescaled
    t_reshape = tf.reshape(t, tf.shape(phie))
    i0_a_reshape = tf.reshape(i0_a, tf.shape(phie))
    resc_phie = self.params["rescale_phie"]
    phie0 = self.params["phie0"](
        i0_a_reshape,
        self.params["j_a"],
        self.params["F"],
        self.params["R"],
        self.params["T"],
        self.params["Uocp_a0"],
    )
    return (
        resc_phie
        * phie
        * (np.float64(1.0) - tf.exp(-t_reshape / self.hard_IC_timescale))
        + phie0
    )


# @conditional_decorator(tf.function,optimized)
def rescalePhis_c(self, phis_c, t, i0_a):
    # t and x inputs have physical units
    # State variables inputs are rescaled
    t_reshape = tf.reshape(t, tf.shape(phis_c))
    i0_a_reshape = tf.reshape(i0_a, tf.shape(phis_c))
    resc_phis_c = self.params["rescale_phis_c"]
    phis_c0 = self.params["phis_c0"](
        i0_a_reshape,
        self.params["j_a"],
        self.params["F"],
        self.params["R"],
        self.params["T"],
        self.params["Uocp_a0"],
        self.params["j_c"],
        self.params["i0_c0"],
        self.params["Uocp_c0"],
    )
    return (
        resc_phis_c
        * phis_c
        * (np.float64(1.0) - tf.exp(-t_reshape / self.hard_IC_timescale))
        + phis_c0
    )


# @conditional_decorator(tf.function,optimized)
def rescaleCs_a(self, cs_a, t):
    # t and x inputs have physical units
    # State variables inputs are rescaled
    t_reshape = tf.reshape(t, tf.shape(cs_a))
    resc_cs_a = self.params["csanmax"]
    return (
        resc_cs_a
        * cs_a
        * (np.float64(1.0) - tf.exp(-t_reshape / self.hard_IC_timescale))
        + tf.exp(-t_reshape / self.hard_IC_timescale) * self.cs_a0
    )


# @conditional_decorator(tf.function,optimized)
def rescaleCs_c(self, cs_c, t):
    # t and x inputs have physical units
    # State variables inputs are rescaled
    t_reshape = tf.reshape(t, tf.shape(cs_c))
    resc_cs_c = self.params["cscamax"]
    return (
        resc_cs_c
        * cs_c
        * (np.float64(1.0) - tf.exp(-t_reshape / self.hard_IC_timescale))
        + tf.exp(-t_reshape / self.hard_IC_timescale) * self.cs_c0
    )
