#!/bin/bash

# Python arguments
# random_state=1220
# random_state_pre=1220
# results_root=/results
# batch_size=512

# Slurm arguments
node=a[001-004,006]
partition=yuxinchen-contrib
mem=11G
jobname=run

# Get the results for the dense network
srun -w ${node} --gres=gpu:4 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 train.py --config_path=configs/config_svhn_resnet_randomsampling.json


# chmod 777 ${results_root}
