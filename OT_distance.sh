#!/bin/bash

node=r004
partition=yuxinchen-contrib
mem=48G
initial=20
dataset='SVHN'
samplesize=50
ot_distance=0
ot_distance_only=0
jobname=OT_$ot_distance_$samplesize_$dataset
epochs=700



srun -w ${node} --gres=gpu:1 -c 64 --ntasks-per-node 1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance --Epochs $epochs --OT_distance_only $ot_distance_only


