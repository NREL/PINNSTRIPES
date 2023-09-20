import numpy as np
from scipy.interpolate import CubicSpline


def sortInput(x, y):
    # Sort input if not sorted
    x = list(x)
    y = list(y)
    y = [val for _, val in sorted(zip(x, y))]
    x.sort()
    return x, y


def generateTFSpline(x, y, filename, funcname, precision="float64", mode="w+"):
    x, y = sortInput(x, y)
    # Get cubic spline
    cs = CubicSpline(x, y)
    f = open(filename, mode)
    if mode == "w+":
        f.write("import argument\n")
        f.write("import numpy as np\n")
        f.write("import tensorflow as tf\n")
        f.write("from conditionalDecorator import conditional_decorator\n")
        f.write("\n")
        f.write(f'tf.keras.backend.set_floatx("{precision}")\n')
        f.write("\n")
        f.write("# Read command line arguments\n")
        f.write("args = argument.initArg()\n")
        f.write("\n")
        f.write("if args.optimized:\n")
        f.write("    optimized = True\n")
        f.write("else:\n")
        f.write("    optimized = False\n")
        f.write("\n")
    elif mode == "a+":
        f.write("\n")
    f.write("@conditional_decorator(tf.function, optimized)\n")
    f.write(f"def {funcname}(x):\n")
    f.write(f"    res = np.{precision}(0)\n")
    for i in range(len(x) - 1):
        # domain {x[i], x[i+1]}
        localCoeffList = "[ "
        for icoeff, coeff in enumerate(list(cs.c[:, i])):
            localCoeffList += f"np.{precision}({coeff})"
            if icoeff == len(cs.c[:, i]) - 1:
                localCoeffList += "]"
            else:
                localCoeffList += ", "
        if i == len(x) - 2:
            f.write(
                f"    condition = tf.math.logical_and(x >= np.{precision}({x[i]}), x <= np.{precision}({x[i+1]}))\n"
            )
        else:
            f.write(
                f"    condition = tf.math.logical_and(x >= np.{precision}({x[i]}), x < np.{precision}({x[i+1]}))\n"
            )
        f.write(
            f"    res += tf.where(condition, tf.math.polyval({localCoeffList}, x - np.{precision}({x[i]})), np.{precision}(0))\n"
        )
    f.write("    return res\n")
    f.close()


def generateTFPoly(coeffs, filename, funcname, precision="float64", mode="w+"):
    coeffs = list(coeffs)
    f = open(filename, mode)
    if mode == "w+":
        f.write("import argument\n")
        f.write("import numpy as np\n")
        f.write("import tensorflow as tf\n")
        f.write("from conditionalDecorator import conditional_decorator\n")
        f.write("\n")
        f.write(f'tf.keras.backend.set_floatx("{precision}")\n')
        f.write("\n")
        f.write("# Read command line arguments\n")
        f.write("args = argument.initArg()\n")
        f.write("\n")
        f.write("if args.optimized:\n")
        f.write("    optimized = True\n")
        f.write("else:\n")
        f.write("    optimized = False\n")
        f.write("\n")
    elif mode == "a+":
        f.write("\n")
    f.write("@conditional_decorator(tf.function, optimized)\n")
    f.write(f"def {funcname}(x):\n")
    listStr = "["
    for icoeff, coeff in enumerate(coeffs):
        listStr += f"np.{precision}({coeff})"
        if icoeff == len(coeffs) - 1:
            listStr += "]"
        else:
            listStr += ", "
    f.write(f"    return tf.math.polyval({listStr}, x)\n")
    f.close()


def generateComsolSpline(x, y, filename, funcname, mode):
    x, y = sortInput(x, y)
    # Get cubic spline
    cs = CubicSpline(x, y)
    f = open(filename, mode)
    f.write(funcname)
    f.write("\n")
    for i in range(len(x) - 1):
        # domain {x[i], x[i+1]}
        polynomialString = f"{cs.c[0,i]}*(x-{x[i]})^3+({cs.c[1,i]})*(x-{x[i]})^2+({cs.c[2,i]})*(x-{x[i]})+({cs.c[3,i]})"
        if i == len(x) - 2:
            f.write(f"if(x<={x[i+1]}&&x>={x[i]},{polynomialString},0)\n")
        else:
            f.write(f"if(x<{x[i+1]}&&x>={x[i]},{polynomialString},0)+")
    f.write("\n")
    f.close()


def generateComsolPoly(coeffs, filename, funcname, mode):
    coeffs = list(coeffs)
    f = open(filename, mode)
    f.write(funcname)
    f.write("\n")
    polyStr = ""
    for icoeff, coeff in enumerate(coeffs):
        if icoeff == len(coeffs) - 1:
            polyStr += f"({coeff})\n"
        elif icoeff == len(coeffs) - 2:
            polyStr += f"({coeff})*x+"
        else:
            polyStr += f"({coeff})*x^{len(coeffs)-1-icoeff}+"
    f.write(polyStr)
    f.write("\n")
    f.close()
