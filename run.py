import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Run the Trainer implementation")

parser.add_argument(
    "--name",
    type=str,
    metavar="NAME",
    help="Name to save the results",
)
parser.add_argument(
    "--percentage",
    type=int,
    metavar="PERCENTAGE",
    help="Percentage of training dataset",
    default=100,
)

args = parser.parse_args()

name = 'conll_wikiner' #args.name
PERC = args.percentage
BATCH_SIZES = [16]
EPOCHS = [2,3]
LENGTHS = [1]
LR = [2e-5, 3e-5]

for bs in BATCH_SIZES:
    for e in EPOCHS:
        for lr in LR:
            output = subprocess.check_output(
                f"python3 wikiner_trainer.py {name}-{bs}-{e}-{lr} \
                                            --batch_size {bs} \
                                            --epochs {e} \
                                            --lr {lr} \
                                            ",
                shell=True,
            )
