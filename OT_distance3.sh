#!/bin/bash

node=c001
partition=yuxinchen-contrib
mem=24G
initial=10
dataset='CIFAR10'
samplesize=100
ot_distance=1
ot_distance_only=0
sigmoid=0
net_trained=20
jobname=OT_distance_$net_trained_$samplesize


srun -w ${node} --gres=gpu:1 -c 48 --ntasks-per-node 1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance_eval.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance --Net_trained $net_trained --OT_distance_only $ot_distance_only --Sigmoid $sigmoid


