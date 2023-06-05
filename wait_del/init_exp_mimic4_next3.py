import torch
import numpy as np
from args.args_mtanivp import ArgsMTANIVP
from args.args_raindrop import ArgsRaindrop
from args.args_ckconv import ArgsCKCONV
from args.args_ivpvae import ArgsIVPVAE
from args.args_mtan import ArgsMTAN

torch.manual_seed(0)
np.random.seed(0)


class Args_Next_M4_Init_LEIT_IVPVAE_ResNetFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_Next_M4_Init_LEIT_IVPVAE_ResNetFlow, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'next'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'testing'
        self.ivp_solver = 'resnetflow'
        self.num_dl_workers = 8
        self.data = 'm4_next_100'
        self.device = 'cuda:1'
        self.train_w_mask = False
