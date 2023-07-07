#!/bin/bash

args_in="--random-state=1;--evolve_module=ivp;--data=p12;--variable-num=41;--num-times=2881;--time-max=2880;--time-scale=none;--ml-task=extrap;--model-type=initialize;--mhatt_n_layer=4;--n_embd=240;--next-headn=10;--patience=10;--log-tool=logging;--device=cuda:1;--test-info=test"

main="/home/xiao/project/leit_gpt/main.py"

args="${args_in//=/ }"
args="${args//;/ }"

export WANDB_MODE=online

~/tools/miniconda3/envs/leit/bin/python $main $args