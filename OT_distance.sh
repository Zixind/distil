#!/bin/bash

node=a004
partition=yuxinchen-contrib
mem=11G
initial=20
dataset='MNIST'
samplesize=20
ot_distance=1
jobname=OT_distance_$samplesize_$dataset



srun -w ${node} --gres=gpu:1 -c 64 --ntasks-per-node 1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 OT_distance.py --Label_Initialize $initial --dataset $dataset --sample_size $samplesize --OT_distance $ot_distance


