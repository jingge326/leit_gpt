#!/bin/bash

args_in="--random-state=1;--evolve_module=ivp;--mhatt_n_layer=4;--n_embd=240;--batch-size=64;--device=cuda:1;--test-info=evolve_ivp"

main="/home/xiao/project/leit_gpt/main.py"

args="${args_in//=/ }"
args="${args//;/ }"

~/tools/miniconda3/envs/leit/bin/python $main $args
