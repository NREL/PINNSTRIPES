import sys


# Adapted from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/34325723#34325723
# Print iterations progress
def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if total < 1e-12:
        return
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\x1b[2K\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()
        sys.stdout.flush()
