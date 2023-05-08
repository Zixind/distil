#!/bin/bash

# Python arguments
# random_state=1220
# random_state_pre=1220
# results_root=/results
# batch_size=512

# Slurm arguments
node=c001    #only a002 working
partition=general
mem=24G
jobname=GLISTER_SVHN

# Get the results for the dense network
srun -w ${node} --gres=gpu:1 -c 48 --ntasks-per-node=1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 classification3.py --acquisition GLISTER --dataset SVHN


# chmod 777 ${results_root}
