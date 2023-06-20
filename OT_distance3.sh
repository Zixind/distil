#!/bin/bash

node=b002
partition=yuxinchen-contrib
mem=48G
initial=10
dataset='SVHN'
samplesize=100
ot_distance=1
ot_distance_only=1
sigmoid=0
net_trained=150
jobname=OT_distance_$net_trained_$samplesize


srun -w ${node} --gres=gpu:1 -c 32 --ntasks-per-node 2 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance_eval.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance --Net_trained $net_trained --OT_distance_only $ot_distance_only --Sigmoid $sigmoid

