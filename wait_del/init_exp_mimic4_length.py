import os

import torch
import numpy as np

from args.args_ivpvae import ArgsIVPVAE

torch.manual_seed(0)
np.random.seed(0)


class Args_Length_M4_Init_LEIT_IVPVAE_ResNetFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_Length_M4_Init_LEIT_IVPVAE_ResNetFlow, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'length'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.reg_input = 'z0'
        self.test_info = 'r2_p5_e41'
        self.ivp_solver = 'resnetflow'
        self.num_dl_workers = 8
        self.data = 'm4_length_100'
        self.device = 'cuda:3'
