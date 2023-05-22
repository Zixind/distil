#!/bin/bash

node=r[004-005]
partition=yuxinchen-contrib
mem=48G
jobname=OT_distance
initial=20
dataset='CIFAR10'
samplesize=80

srun -w ${node} --gres=gpu:2 -c 32 --ntasks-per-node 2 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize


