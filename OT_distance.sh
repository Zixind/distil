#!/bin/bash

node=f[002-003]
partition=yuxinchen-contrib
mem=48G
jobname=OT_distance_net_120
initial=20
dataset='CIFAR10'
samplesize=120
ot_distance=1


srun -w ${node} --gres=gpu:2 -c 32 --ntasks-per-node 1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance


