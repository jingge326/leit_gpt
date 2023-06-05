import os

import torch
import numpy as np

from args.args_ivpvae import ArgsIVPVAE
from args.args_mtanivp import ArgsMTANIVP

torch.manual_seed(0)
np.random.seed(0)

pretrained_models = os.getcwd() + '/temp/pl_checkpoint/'

finetune_models = os.getcwd() + '/temp/model/'


class Args_FT_Mortality_M4_Reconst_LEIT_IVPVAE_ODE(ArgsIVPVAE):
    def __init__(self):
        super(Args_FT_Mortality_M4_Reconst_LEIT_IVPVAE_ODE,
              self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'biclass'
        self.target = 'representation'
        self.model_type = 'reconstruct'  # ['initialize', 'reconstruct']
        self.former_ehr_variable_num = 96
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = False
        self.test_info = 'pr4_p5_e53'

        self.ivp_solver = 'ode'
        # For ODE
        self.activation
        self.final_activation
        self.num_dl_workers = 8

        self.load_para_path = pretrained_models + \
            'ivp_vae_pretrain_data_mimic4_pretraining_40k_r4_ode_pretrain_initialize_mlp_unfreeze_True__epoch=53-val_loss=733.00.ckpt'
        self.data = 'm4_mortality_100'
        self.device = 'cuda:3'


class Args_FT_Mortality_M4_Reconst_LEIT_MTANIVP(ArgsMTANIVP):
    def __init__(self):
        super(Args_FT_Mortality_M4_Reconst_LEIT_MTANIVP,
              self).__init__()
        self.leit_model = 'mtan_ivp'
        self.task = 'biclass'
        self.target = 'representation'  # ['scratch', 'representation']
        self.model_type = 'reconstruct'  # ['initialize', 'reconstruct']
        self.former_ehr_variable_num = 96
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = False
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        self.ivp_solver = 'resnetflow'
        self.ratio_ce = 1000
        self.test_info = 'r4_p5_e89'
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:2'
        self.batch_size = 64
        self.load_para_path = pretrained_models + \
            'mtan_ivp_pretrain_data_mimic4_pretraining_40k_r4_pretrain_initialize_unfreeze_True_r4_p5_epoch=89-val_loss=628.45.ckpt'


class Args_FT_Mortality_M4_Reconst_LEIT_IVPVAE_ResNetFlow_P12(ArgsIVPVAE):
    def __init__(self):
        super(Args_FT_Mortality_M4_Reconst_LEIT_IVPVAE_ResNetFlow_P12,
              self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'biclass'
        self.target = 'representation'  # ['scratch', 'representation']
        self.model_type = 'reconstruct'  # ['initialize', 'reconstruct']
        self.former_ehr_variable_num = 96
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = False
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        self.ivp_solver = 'resnetflow'
        self.ratio_ce = 1000
        self.test_info = ''
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:2'
        self.batch_size = 64
        self.load_para_path = pretrained_models + \
            'mtan_ivp_pretrain_data_mimic4_pretraining_40k_r4_pretrain_initialize_unfreeze_True_r4_p5_epoch=89-val_loss=628.45.ckpt'
        self.twin_flow = True
        self.train_w_mask = False
        self.quantization = 0.0167
        self.n_samples = 8000
        self.extrap = False
