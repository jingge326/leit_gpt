import os

import torch
import numpy as np

from args.args_ivpvae import ArgsIVPVAE

torch.manual_seed(0)
np.random.seed(0)

pretrained_models = os.getcwd() + '/temp/pl_checkpoint/'
finetune_models = os.getcwd() + '/temp/model/'


class Args_FT_Length_M4_Reconst_LEIT_IVPVAE_ResNetFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_FT_Length_M4_Reconst_LEIT_IVPVAE_ResNetFlow, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'length'
        self.target = 'representation'  # ['scratch', 'representation']
        self.model_type = 'reconstruct'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.former_ehr_variable_num = 96
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = False
        self.reg_input = 'z0'
        self.test_info = 'r2_p5_e41'
        self.ivp_solver = 'resnetflow'
        self.num_dl_workers = 8
        self.load_para_path = pretrained_models + \
            'pretrain_data_mimic4_pretraining_40k_r4_grd_vae_resnetflow_pretrain_initialize_mlp_unfreeze_True__epoch=41-val_loss=742.69.ckpt'
        self.data = 'm4_length_100'
        self.device = 'cuda:1'
