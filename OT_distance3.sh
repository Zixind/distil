#!/bin/bash

node=c001
partition=yuxinchen-contrib
mem=24G
initial=20
dataset='MNIST'
samplesize=100
ot_distance=1
ot_distance_only=1
net_trained=50
jobname=OT_distance_$net_trained_$samplesize


srun -w ${node} --gres=gpu:1 -c 16 --ntasks-per-node 3 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance3.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance --Net_trained $net_trained --OT_distance_only $ot_distance_only


