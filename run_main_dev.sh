#!/bin/bash

args_in="--random-state=1;--leit-model=att_ivp_vae;--ml-task=extrap;--time-scale=constant;--time-constant=2880;--data=p12;--variable-num=41;--time-max=2880;--combine-methods=attn_init;--test-info=mulattn_init;--device=cuda:3"

main="/home/xiao/project/leit_dev/main.py"

args="${args_in//=/ }"
args="${args//;/ }"

~/.conda/envs/leit/bin/python $main $args