#!/bin/bash

node=aa002
partition=yuxinchen-contrib
mem=8G
jobname=OT_distance_evaluate
initial=20
dataset='CIFAR10'
samplesize=20
ot_distance=1
net_trained=100

srun -w ${node} --gres=gpu:1 -c 32 --ntasks-per-node 1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance3.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance --Net_trained $net_trained

