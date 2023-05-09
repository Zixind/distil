#!/bin/bash

# Python arguments
# random_state=1220
# random_state_pre=1220
# results_root=/results
# batch_size=512

# Slurm arguments
node=aa[001-002]   #only a002 working
partition=yuxinchen-contrib
mem=8G
jobname=GLISTER_MNIST

# Get the results for the dense network
srun -w ${node} --gres=gpu:2 -c 32 --ntasks-per-node=1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 classification3.py --acquisition GLISTER --dataset MNIST


# chmod 777 ${results_root}
