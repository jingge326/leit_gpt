#!/bin/bash


args_in="--random-state=1;--mhatt_n_layer=4;--n_embd=240;--batch-size=64;--test-info=evolve_delta"

main="/home/xiao/project/leit_gpt/main.py"

args="${args_in//=/ }"
args="${args//;/ }"

~/tools/miniconda3/envs/leit/bin/python $main $args