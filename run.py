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

name = args.name
PERC = args.percentage
BATCH_SIZES = [16]
EPOCHS = [3, 4, 5]
LENGTHS = [128, 256]
LR = [1e-5, 2e-5, 3e-5]


for l in LENGTHS:
    for bs in BATCH_SIZES:
        for e in EPOCHS:
            for lr in LR:
                output = subprocess.check_output(
                    f"python3 trainer.py {name}-{bs}-{e}-{lr}-{l} \
                                                --batch_size {bs} \
                                                --epochs {e} \
                                                --max_len {l} \
                                                --lr {lr} \
                                                --percentage {PERC} \
                                                ",
                    shell=True,
                )
                #with open(f"results/{name}-{bs}-{e}-{lr}-{l}", "+a") as f:
                #    f.write(output.decode())
