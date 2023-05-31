#!/bin/bash

node=d001
partition=yuxinchen-contrib
mem=48G
jobname=OT_distance_net
initial=20
dataset='CIFAR10'
samplesize=50

srun -w ${node} --gres=gpu:1 -c 64 --ntasks-per-node 1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize


