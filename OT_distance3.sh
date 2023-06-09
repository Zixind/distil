#!/bin/bash

node=j003
partition=yuxinchen-contrib
mem=48G
initial=20
dataset='SVHN'
samplesize=100
ot_distance=0
ot_distance_only=0
net_trained=50
jobname=OT_distance_$net_trained_$samplesize


srun -w ${node} --gres=gpu:1 -c 32 --ntasks-per-node 2 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance3.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance --Net_trained $net_trained --OT_distance_only $ot_distance_only


