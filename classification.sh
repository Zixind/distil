#!/bin/bash

# Python arguments
# random_state=1220
# random_state_pre=1220
# results_root=/results
# batch_size=512

# Slurm arguments
node=e002,r[001,003]
partition=yuxinchen-contrib
mem=48G
jobname=SVHN_CoreSet
arguments="--dataset SVHN --acquisition CoreSet"

# Get the results for the dense network
srun -w ${node} --gres=gpu:3 -c 64 --ntasks-per-node=1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 -m classification3 $arguments


# chmod 777 ${results_root}
