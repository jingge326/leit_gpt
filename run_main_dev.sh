#!/bin/bash

args_in="--random-state=1;--data=p12;--variable-num=37;--time-max=2880;--time-scale=none;--ml-task=interp;--model-type=reconstruct;--pre_model=bert_pretrain_initialize_nhead12_nlyrs4_bsize32_mm_cls_r1.pt;--train_obj=bert;--add_cls;--gpts_output=cls;--t-offset=1.0;--times-drop=0.1;--del_bad_p12;--evolve_module=ivp;--mhatt_n_layer=4;--n_embd=240;--batch-size=32;--patience=10;--test-info=v37_bert_cls"

main="/home/xiao/project/leit_gpt/main.py"

args="${args_in//=/ }"
args="${args//;/ }"

export WANDB_MODE=online

~/tools/miniconda3/envs/leit/bin/python $main $args