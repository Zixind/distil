#!/bin/bash

node=a004
partition=yuxinchen-contrib
mem=11G
initial=20
dataset='CIFAR10'
samplesize=10
ot_distance=1
net_trained=50
jobname=OT_distance_$net_trained_$samplesize


srun -w ${node} --gres=gpu:1 -c 32 --ntasks-per-node 2 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance3.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance --Net_trained $net_trained


