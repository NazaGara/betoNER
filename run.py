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
    "--trainer",
    type=str,
    metavar="TRAINER",
    help="trainer to use",
    default='conll',
)

args = parser.parse_args()

name = args.name
TRAINER = args.trainer
BATCH_SIZES = [16]
EPOCHS = [2,3,4]
LR = [1e-5, 2e-5, 3e-5]

for bs in BATCH_SIZES:
    for e in EPOCHS:
        for lr in LR:
            output = subprocess.check_output(
                f"python3 {TRAINER}_trainer.py {name}-{bs}-{e}-{lr} \
                                            --batch_size {bs} \
                                            --epochs {e} \
                                            --lr {lr} \
                                            ",
                shell=True,
            )
