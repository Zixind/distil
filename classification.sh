#!/bin/bash

# Python arguments
# random_state=1220
# random_state_pre=1220
# results_root=/results
# batch_size=512

# Slurm arguments
node=r[001-005]
partition=yuxinchen-contrib
mem=48G
jobname=SVHN_CoreSet_250_200
arguments="--dataset SVHN --acquisition CoreSet --batch_size 250 --Label_Initialize 200"

# Get the results for the dense network
srun -w ${node} --gres=gpu:4 -c 32 --ntasks-per-node=1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 -m classification $arguments


# chmod 777 ${results_root}
