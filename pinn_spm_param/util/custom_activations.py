import argument
import numpy as np
import tensorflow as tf
from conditionalDecorator import conditional_decorator
from keras.backend import set_floatx
from keras.layers import Activation, Layer
from keras.utils import get_custom_objects

set_floatx("float64")

# Read command line arguments
args = argument.initArg()

if args.optimized:
    optimized = True
else:
    optimized = False


@conditional_decorator(tf.function, optimized)
def swish_activation(x):
    """
    Swish activation - with beta not-traininable!

    """
    return x * tf.math.sigmoid(x)


class Bswish(Layer):
    def __init__(self, num_outputs):
        super(Bswish, self).__init__()
        self.num_outputs = num_outputs
        beta_init = tf.random_normal_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=(1, num_outputs), dtype="float64"),
            trainable=True,
            name="beta",
        )
        b_init = tf.random_normal_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(1, num_outputs), dtype="float64"),
            trainable=True,
            name="bias",
        )

    def call(self, inputs):
        return inputs * tf.math.sigmoid(inputs * self.beta) + self.b


get_custom_objects().update({"swish_activation": Activation(swish_activation)})
