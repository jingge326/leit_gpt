#!/bin/bash

# args_in="--random-state=1;--data=p12;--variable-num=41;--time-max=1439;--num-times=1440;--ml-task=biclass;--model-type=initialize;--mhatt_n_layer=4;--n_embd=240;--bias;--gpts_output=last;--batch-size=64;--test-info=last"

# main="/home/xiao/project/leit_gpt/main.py"

# args="${args_in//=/ }"
# args="${args//;/ }"

# ~/tools/miniconda3/envs/leit/bin/python $main $args


args_in="--random-state=2;--data=p12;--variable-num=41;--time-max=1439;--num-times=1440;--ml-task=biclass;--model-type=initialize;--mhatt_n_layer=4;--n_embd=240;--bias;--gpts_output=last;--batch-size=64;--test-info=last"

main="/home/xiao/project/leit_gpt/main.py"

args="${args_in//=/ }"
args="${args//;/ }"

~/tools/miniconda3/envs/leit/bin/python $main $args



args_in="--random-state=3;--data=p12;--variable-num=41;--time-max=1439;--num-times=1440;--ml-task=biclass;--model-type=initialize;--mhatt_n_layer=4;--n_embd=240;--bias;--gpts_output=last;--batch-size=64;--test-info=last"

main="/home/xiao/project/leit_gpt/main.py"

args="${args_in//=/ }"
args="${args//;/ }"

~/tools/miniconda3/envs/leit/bin/python $main $args