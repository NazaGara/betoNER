import os
import subprocess

BATCH_SIZES = [8, 16]
EPOCHS = [3, 4, 5]
LR = [1e-5, 2e-5, 3e-5]
PERC = 7


for bs in BATCH_SIZES:
    for e in EPOCHS:
        for lr in LR:
            output = subprocess.check_output(
                f"python3 trainer.py overfit-{bs}-{e}-{lr} \
                                            --train_batch_size {bs} \
                                            --epochs {e} \
                                            --lr {lr} \
                                            --percentage {PERC} \
                                            ",
                shell=True,
            )
            with open("results/overfit-{bs}-{e}-{lr}", "+a") as f:
                f.wirte(output.decode())
