import sys
import time

from main import *

n_repeat = 40


def main_repeat(id_sim, args, input_params):
    # Read command line arguments
    input_params["ID"] = id_sim
    input_params["seed"] = -1
    nn = initialize_nn(args=args, input_params=input_params)
    elapsed, unweighted_loss = do_training(input_params=input_params, nn=nn)
    return elapsed, unweighted_loss


args = argument.initArg()
input_params = initialize_params(args)


for i_repeat in range(n_repeat):
    print(f"\n\nNREPEAT {i_repeat+1}/{n_repeat}")
    elapsed, unweighted_loss = main_repeat(i_repeat, args, input_params)
    print(f"Elapsed time {elapsed:.2f}s")
    print(f"Unweighted loss {unweighted_loss}")
