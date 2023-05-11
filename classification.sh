#!/bin/bash

# Python arguments
# random_state=1220
# random_state_pre=1220
# results_root=/results
# batch_size=512

# Slurm arguments
node=b002,e002,f[002-003]
partition=yuxinchen-contrib
mem=48G
jobname=CIFAR10_random_200_1000
arguments="--dataset CIFAR10 --acquisition random --batch_size 200 --Label_Initialize 1000"

# Get the results for the dense network
srun -w ${node} --gres=gpu:4 -c 16 --ntasks-per-node=2 --mem ${mem} -p ${partition} --job-name=${jobname} python3 -m classification $arguments


# chmod 777 ${results_root}
