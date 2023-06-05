import os

import torch
import numpy as np
from args.args_mtan import ArgsMTAN
from args.args_mtanivp import ArgsMTANIVP
from args.args_ivpvae import ArgsIVPVAE


import sys
sys.setrecursionlimit(1500)

torch.manual_seed(0)
np.random.seed(0)

pretrained_models = os.getcwd() + '/temp/pl_checkpoint/'


class Args_Pretrain_M4_Init_LEIT_ResNetFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_Pretrain_M4_Init_LEIT_ResNetFlow,
              self).__init__()
        self.task = 'pretrain'
        self.target = 'pretrain'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = ''
        self.ivp_solver = 'resnetflow'
        self.embedding_method = 'mlp'
        self.gpus = [0]    # or -1
        self.data = 'data_mimic4_pretraining_40k_r4'
        self.dev_mode = 'run'  # ['run', 'debug']
        self.num_dl_workers = 4


class Args_Pretrain_M4_Init_LEIT_ODE(ArgsIVPVAE):
    def __init__(self):
        super(Args_Pretrain_M4_Init_LEIT_ODE, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'pretrain'
        self.target = 'pretrain'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = ''
        self.ivp_solver = 'ode'
        self.embedding_method = 'mlp'
        self.gpus = [0, 1, 2, 3]    # or -1
        self.data = 'data_mimic4_pretraining_40k_r4'
        self.dev_mode = 'run'  # ['run', 'debug']
        self.num_dl_workers = 4


class Args_Pretrain_M4_Init_LEIT_GRU(ArgsIVPVAE):
    def __init__(self):
        super(Args_Pretrain_M4_Init_LEIT_GRU,
              self).__init__()
        self.leit_model = 'classic_rnn'
        self.cell = 'gru'    # ['gru', 'expdecay']
        self.task = 'pretrain'
        self.target = 'pretrain'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = ''
        self.gpus = [0, 2, 3]    # or -1
        self.data = 'data_mimic4_pretraining_40k_r4'
        self.dev_mode = 'run'  # ['run', 'debug']
        self.num_dl_workers = 4


class Args_Pretrain_M4_Init_LEIT_GRUD(ArgsIVPVAE):
    def __init__(self):
        super(Args_Pretrain_M4_Init_LEIT_GRUD,
              self).__init__()
        self.leit_model = 'classic_rnn'
        self.cell = 'expdecay'    # ['gru', 'expdecay']
        self.task = 'pretrain'
        self.target = 'pretrain'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct', 'load']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = ''
        self.gpus = [0, 1, 2, 3]    # or -1
        self.data = 'data_mimic4_pretraining_40k_r4'
        self.dev_mode = 'run'  # ['run', 'debug']
        self.num_dl_workers = 8


class Args_Pretrain_M4_Load_LEIT_GRUD(ArgsIVPVAE):
    def __init__(self):
        super(Args_Pretrain_M4_Load_LEIT_GRUD,
              self).__init__()
        self.leit_model = 'classic_rnn'
        self.cell = 'expdecay'    # ['gru', 'expdecay']
        self.task = 'pretrain'
        self.target = 'pretrain'
        self.model_type = 'load'  # ['initialize', 'reconstruct', 'load']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = ''
        self.gpus = [0, 1, 2, 3]    # or -1
        self.data = 'data_mimic4_pretraining_40k_r4'
        self.dev_mode = 'debug'  # ['run', 'debug']
        self.num_dl_workers = 8
        self.load_para_path = pretrained_models + \
            'classic_rnn_pretrain_data_mimic4_pretraining_40k_r4_expdecay_pretrain_initialize_mlp_unfreeze_True__epoch=17-val_loss=1228.11.ckpt'


class Args_Pretrain_M4_Init_LEIT_MTAN(ArgsMTAN):
    def __init__(self):
        super(Args_Pretrain_M4_Init_LEIT_MTAN,
              self).__init__()
        self.leit_model = 'mtan'
        self.task = 'pretrain'
        self.target = 'pretrain'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'b16'
        self.gpus = [0, 1, 2, 3]    # or -1
        self.data = 'data_mimic4_pretraining_40k_r4'
        self.dev_mode = 'run'  # ['run', 'debug']
        self.num_dl_workers = 8
        self.batch_size = 16


class Args_Pretrain_M4_Init_LEIT_MTANIVP(ArgsMTANIVP):
    def __init__(self):
        super(Args_Pretrain_M4_Init_LEIT_MTANIVP, self).__init__()
        self.leit_model = 'mtan_ivp'
        self.task = 'pretrain'
        self.target = 'pretrain'  # ['scratch', 'representation']
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.ivp_solver = 'resnetflow'
        self.test_info = 'r4_p5'
        self.num_dl_workers = 8
        self.gpus = [0, 1, 2, 3]    # or -1
        self.data = 'data_mimic4_pretraining_40k_r4'
        self.dev_mode = 'run'  # ['run', 'debug']
        self.batch_size = 64
