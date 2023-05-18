#!/bin/bash

# Python arguments
# random_state=1220
# random_state_pre=1220
# results_root=/results
# batch_size=512

# Slurm arguments
node=c001 
partition=yuxinchen-contrib
mem=24G
jobname=CIFAR10_random_500_200
arguments="--dataset CIFAR10 --acquisition random --batch_size 500 --Label_Initialize 200"

# Get the results for the dense network
srun -w ${node} --gres=gpu:1 -c 16 --ntasks-per-node=3 --mem ${mem} -p ${partition} --job-name=${jobname} python3 -m classification $arguments


# chmod 777 ${results_root}
