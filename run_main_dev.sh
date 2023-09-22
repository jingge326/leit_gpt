#!/bin/bash

args_in="--random-state=1;--leit-model=ivpattn;--ml-task=extrap;--data=synthetic;--ml-task=syn_extrap;--hidden-layers=1;--log-tool=logging;--variable-num=1;--nhead=8;--test-info=testing"

main="/home/xiao/project/leit_gpt/main.py"

args="${args_in//=/ }"
args="${args//;/ }"

export WANDB_MODE=online

~/tools/miniconda3/envs/leit/bin/python $main $args