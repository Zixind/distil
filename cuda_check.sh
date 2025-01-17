#!/bin/bash

# Python arguments
# random_state=1220
# random_state_pre=1220
# results_root=/results
# batch_size=512

# Slurm arguments
node=f[002-003],r003    
partition=yuxinchen-contrib
mem=48G
jobname=cuda_check

# Get the results for the dense network
srun -w ${node} --gres=gpu:4 -c 6 --mem ${mem} -p ${partition} --job-name=${jobname} python3 cuda_check.py


# chmod 777 ${results_root}
