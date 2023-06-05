import torch
import numpy as np
from args.args_mtanivp import ArgsMTANIVP
from args.args_raindrop import ArgsRaindrop
from args.args_ckconv import ArgsCKCONV
from args.args_ivpvae import ArgsIVPVAE
from args.args_mtan import ArgsMTAN
from args.args_redvae import ArgsREDVAE

torch.manual_seed(0)
np.random.seed(0)


class Args_P19_Init_LEIT_CKCONV(ArgsCKCONV):
    def __init__(self):
        super(Args_P19_Init_LEIT_CKCONV, self).__init__()
        self.leit_model = 'ckconv'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'testing'
        self.num_dl_workers = 8
        self.device = 'cuda:0'
        self.batch_size = 64
        self.no_blocks = 2
        self.data = 'P19'
        self.clip_gradient = False
        self.n_times = 72


class Args_P19_Init_LEIT_IVPVAE_ResNetFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_P19_Init_LEIT_IVPVAE_ResNetFlow, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'ra001'
        self.ivp_solver = 'resnetflow'
        self.ratio_ce = 100
        self.num_dl_workers = 8
        self.data = 'P19'
        self.device = 'cuda:1'
        self.train_w_mask = True
        self.mask_loss_ratio = 0.01
        self.n_times = 72
        self.variable_num = 34


class Args_P19_Init_LEIT_IVPVAE_ResNetFlow_noM(ArgsIVPVAE):
    def __init__(self):
        super(Args_P19_Init_LEIT_IVPVAE_ResNetFlow_noM, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'm_false'
        self.ivp_solver = 'resnetflow'
        self.ratio_ce = 100
        self.num_dl_workers = 8
        self.data = 'P19'
        self.device = 'cuda:3'
        self.train_w_mask = False
        self.n_times = 72
        self.variable_num = 34


class Args_P19_Init_LEIT_REDVAE_ResNetFlow(ArgsREDVAE):
    def __init__(self):
        super(Args_P19_Init_LEIT_REDVAE_ResNetFlow, self).__init__()
        self.leit_model = 'red_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = ''
        self.ivp_solver = 'resnetflow'
        self.ratio_ce = 100
        self.num_dl_workers = 8
        self.data = 'P19'
        self.device = 'cuda:1'
        self.n_times = 72
        self.variable_num = 34


class Args_P19_Init_LEIT_REDVAE_ODE(ArgsREDVAE):
    def __init__(self):
        super(Args_P19_Init_LEIT_REDVAE_ODE, self).__init__()
        self.leit_model = 'red_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = ''
        self.ivp_solver = 'ode'
        self.ratio_ce = 100
        self.num_dl_workers = 8
        self.data = 'P19'
        self.device = 'cuda:2'
        self.n_times = 72
        self.variable_num = 34
        # For ODE
        self.activation = 'ELU'
        self.final_activation = 'Tanh'
