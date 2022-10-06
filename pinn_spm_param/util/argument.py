import argparse


def initArg():
    # CLI
    parser = argparse.ArgumentParser(description="Train PINN")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        metavar="",
        required=False,
        help="Input file for parameters",
        default="input",
    )
    parser.add_argument(
        "-dl",
        "--data_list",
        nargs="+",
        help="<Required> Set flag",
        default=None,
        required=False,
    )
    parser.add_argument(
        "-p",
        "--params_list",
        nargs="+",
        help="List of parameter values (0: i0_a, 1: ds_c)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-df",
        "--dataFolder",
        type=str,
        metavar="",
        required=False,
        help="Folder where comsol data is stored",
        default=None,
    )
    parser.add_argument(
        "-opt",
        "--optimized",
        action="store_true",
        help="Use tf.function for optimization",
    )
    parser.add_argument(
        "-nt",
        "--n_t",
        type=int,
        metavar="",
        required=False,
        help="Number of points along the time axis",
        default=100,
    )
    parser.add_argument(
        "-nx",
        "--n_x",
        type=int,
        metavar="",
        required=False,
        help="Number of points along the X axis",
        default=100,
    )
    parser.add_argument(
        "-nr",
        "--n_r",
        type=int,
        metavar="",
        required=False,
        help="Number of points along the R axis",
        default=100,
    )
    parser.add_argument(
        "-gf",
        "--generatedDataFolder",
        type=str,
        metavar="",
        required=False,
        help="Folder where generated data is stored",
        default="generatedData",
    )
    parser.add_argument(
        "-mf",
        "--modelFolder",
        type=str,
        metavar="",
        required=False,
        help="Folder where model weights are stored",
        default="../Model",
    )
    parser.add_argument(
        "-lf",
        "--logFolder",
        type=str,
        metavar="",
        required=False,
        help="Folder where loss logs are stored",
        default="../Log",
    )
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument(
        "-b", "--big", action="store_true", help="big network version"
    )
    group1.add_argument(
        "-sm", "--small", action="store_true", help="small network version"
    )
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument(
        "-m", "--merged", action="store_true", help="merged network version"
    )
    group2.add_argument(
        "-sp", "--split", action="store_true", help="split network version"
    )
    group3 = parser.add_mutually_exclusive_group()
    group3.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="execute without plotting on screen",
    )
    group3.add_argument(
        "-v", "--verbose", action="store_true", help="plot on screen"
    )
    parser.add_argument(
        "-frt",
        "--frequencyDownsamplingT",
        type=int,
        metavar="",
        required=False,
        help="Frequency at which we downsample time",
        default=1,
    )
    parser.add_argument(
        "-frx",
        "--frequencyDownsamplingX",
        type=int,
        metavar="",
        required=False,
        help="Frequency at which we downsample x",
        default=1,
    )
    parser.add_argument(
        "-frr",
        "--frequencyDownsamplingR",
        type=int,
        metavar="",
        required=False,
        help="Frequency at which we downsample r",
        default=1,
    )
    group4 = parser.add_mutually_exclusive_group()
    group4.add_argument(
        "-simp",
        "--simpleModel",
        action="store_true",
        help="Use simple P2D model",
    )
    group4.add_argument(
        "-fast",
        "--fastModel",
        action="store_true",
        help="Use simple fast P2D model",
    )

    args = parser.parse_args()

    return args
