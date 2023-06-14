#!/bin/bash

node=aa001
partition=yuxinchen-contrib
mem=11G
initial=20
dataset='SVHN'
samplesize=10
ot_distance=0
ot_distance_only=0
sigmoid=0
net_trained=10
jobname=OT_distance_$net_trained_$samplesize


srun -w ${node} --gres=gpu:1 -c 32 --ntasks-per-node 1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance3.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance --Net_trained $net_trained --OT_distance_only $ot_distance_only --Sigmoid $sigmoid


