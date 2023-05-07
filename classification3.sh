#!/bin/bash


# Slurm arguments
node=b[002-003],d[001-002]
partition=yuxinchen-contrib
mem=48G
jobname=GLISTER_CIFAR10

srun -w ${node} --gres=gpu:4 -c 64 --ntasks-per-node=1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 classification3.py --dataset CIFAR10 --acquisition GLISTER



