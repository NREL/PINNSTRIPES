import argument
import numpy as np

import keras
import tensorflow as tf
from conditionalDecorator import conditional_decorator

keras.backend.set_floatx("float64")

# Read command line arguments
args = argument.initArg()

if args.optimized:
    optimized = True
else:
    optimized = False


@conditional_decorator(tf.function, optimized)
def uocp_a_fun_x(x):
    return tf.math.polyval(
        [
            np.float64(1878.6244900261463),
            np.float64(-4981.580023016213),
            np.float64(516.2941996957871),
            np.float64(6452.38177755237),
            np.float64(-436.0524457974526),
            np.float64(1264.0514576769442),
            np.float64(-20918.656956191975),
            np.float64(12954.334261316431),
            np.float64(28871.72866007402),
            np.float64(-37943.83286204571),
            np.float64(34.11141793217983),
            np.float64(29363.16490602074),
            np.float64(-25774.496334571464),
            np.float64(11073.868226559767),
            np.float64(-2702.638445370805),
            np.float64(375.62895901410747),
            np.float64(-28.064663950113868),
            np.float64(1.1265244540945243),
        ],
        x,
    )


@conditional_decorator(tf.function, optimized)
def uocp_c_fun_x(x):
    return tf.math.polyval(
        [
            np.float64(-43309.69063512314),
            np.float64(122888.63938515769),
            np.float64(-69735.99554716503),
            np.float64(-59749.183217994185),
            np.float64(25744.002733171154),
            np.float64(15730.398058573825),
            np.float64(54021.915506318735),
            np.float64(-44566.03206954511),
            np.float64(64.32177924593454),
            np.float64(-7780.173422833786),
            np.float64(1117.4042221859695),
            np.float64(7387.492376558274),
            np.float64(-7237.289515884936),
            np.float64(-705.4465901574707),
            np.float64(17170.20236584321),
            np.float64(-42.60228181558803),
            np.float64(-23266.56994359366),
            np.float64(10810.92851132453),
            np.float64(2545.4065429021307),
            np.float64(1.6554268823619098),
            np.float64(751.3515882152476),
            np.float64(-4447.12851190078),
            np.float64(3727.268889820381),
            np.float64(-1331.1791971457515),
            np.float64(227.4712483170547),
            np.float64(-17.646894926746256),
            np.float64(0.8568207255402533),
            np.float64(-2.34505930698951),
            np.float64(5.059010555584711),
        ],
        x,
    )
