#!/bin/bash

wandb online
export WANDB_MODE=online

args_in="--random-state=1;--train_obj=bert;--evolve_module=ivp;--mhatt_n_layer=4;--n_embd=240;--batch-size=32;--patience=20;--seq_len_min=6;--seq_len_max=200;--times-drop=0.15;--device=cuda:1;--test-info=min7"

main="/home/xiao/project/leit_gpt/main.py"

args="${args_in//=/ }"
args="${args//;/ }"

export WANDB_MODE=online

~/tools/miniconda3/envs/leit/bin/python $main $args