#!/bin/bash


# Slurm arguments
node=r[001-005]
partition=yuxinchen-contrib
mem=48G
jobname=BADGE_SVHN

srun -w ${node} --gres=gpu:4 -c 64 --ntasks-per-node=1 --mem ${mem} -p ${partition} --job-name=${jobname} python3 classification3.py --dataset SVHN --acquisition BADGE



