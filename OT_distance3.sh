#!/bin/bash

node=r002
partition=yuxinchen-contrib
mem=48G
jobname=OT_distance_evaluate
initial=20
dataset='CIFAR10'
samplesize=30

srun -w ${node} --gres=gpu:1 -c 64 --ntasks-per-node 1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance3.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize


