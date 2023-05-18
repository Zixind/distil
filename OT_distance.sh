#!/bin/bash

# Python arguments
# random_state=1220
# random_state_pre=1220
# results_root=/results
# batch_size=512

node=r[003,005]
partition=yuxinchen-contrib
mem=48G
jobname=OT_distance
initial=20
dataset='CIFAR10'
samplesize=80

srun -w ${node} --gres=gpu:2 -c 64 --ntasks-per-node 1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize


