import os
import sys


def parseInputFile(input_filename):
    if not os.path.isfile(input_filename):
        print("ERROR: No input file name found, assuming it is 'input'")
        sys.exit()

    # ~~~~ Parse input
    inpt = {}
    f = open(input_filename)
    data = f.readlines()
    for line in data:
        if ":" in line:
            key, value = line.split(":")
            inpt[key.strip()] = value.strip()
    f.close()

    return inpt
