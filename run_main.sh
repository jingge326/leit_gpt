#!/bin/bash

main="/home/xiao/project/leit_gpt/main.py"

args="${1//=/ }"
args="${args//;/ }"

~/.conda/envs/leit/bin/python $main $args
