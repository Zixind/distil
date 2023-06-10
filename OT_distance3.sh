#!/bin/bash

node=j[002-004]
partition=yuxinchen-contrib
mem=48G
initial=20
dataset='MNIST'
samplesize=100
ot_distance=1
ot_distance_only=0
net_trained=100
jobname=OT_distance_$net_trained_$samplesize


srun -w ${node} --gres=gpu:3 -c 32 --ntasks-per-node 2 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance3.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance --Net_trained $net_trained --OT_distance_only $ot_distance_only


