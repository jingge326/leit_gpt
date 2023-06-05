import torch
import numpy as np
from args.args_mtanivp import ArgsMTANIVP
from args.args_raindrop import ArgsRaindrop
from args.args_ckconv import ArgsCKCONV
from args.args_redvae import ArgsREDVAE
from args.args_mtan import ArgsMTAN
from args.args_ivpvae import ArgsIVPVAE

torch.manual_seed(0)
np.random.seed(0)


class Args_Mortality_P12_Init_LEIT_IVPVAE_ResNetFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_Mortality_P12_Init_LEIT_IVPVAE_ResNetFlow, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'testing'
        self.ivp_solver = 'resnetflow'
        self.num_dl_workers = 8
        self.device = 'cuda:0'
        self.data = 'P12'
        self.quantization = 0.0167    # scale to 1 min interval
        self.n_samples = 8000
        self.extrap = False
        self.variable_num = 41
        self.p12_classify = True
        self.train_w_mask = False
        self.ratio_ce = 100


class Args_Mortality_P12_Init_LEIT_IVPVAE_ResNetFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_Mortality_P12_Init_LEIT_IVPVAE_ResNetFlow, self).__init__()
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
        self.num_dl_workers = 8
        self.device = 'cuda:3'
        self.data = 'P12'
        self.quantization = 0.0167    # scale to 1 min interval
        self.n_samples = 8000
        self.extrap = False
        self.variable_num = 41
        self.p12_classify = True
        self.ratio_ce = 100
        self.train_w_mask = True
        self.mask_loss_ratio = 0.01


class Args_Mortality_P12_Init_LEIT_REDVAE_ResNetFlow(ArgsREDVAE):
    def __init__(self):
        super(Args_Mortality_P12_Init_LEIT_REDVAE_ResNetFlow, self).__init__()
        self.leit_model = 'red_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = ''
        self.ivp_solver = 'resnetflow'
        self.num_dl_workers = 8
        self.device = 'cuda:2'
        self.data = 'P12'
        self.quantization = 0.0167    # scale to 1 min interval
        self.n_samples = 8000
        self.extrap = False
        self.variable_num = 41
        self.p12_classify = True
        self.ratio_ce = 100


class Args_Mortality_P12_Init_LEIT_REDVAE_ODE(ArgsREDVAE):
    def __init__(self):
        super(Args_Mortality_P12_Init_LEIT_REDVAE_ODE, self).__init__()
        self.leit_model = 'red_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'testing'
        self.ivp_solver = 'ode'
        self.num_dl_workers = 8
        self.device = 'cuda:2'
        self.data = 'P12'
        self.quantization = 0.0167    # scale to 1 min interval
        self.n_samples = 8000
        self.extrap = False
        self.variable_num = 41
        self.p12_classify = True
        self.ratio_ce = 100
        # For ODE
        self.activation = 'ELU'
        self.final_activation = 'Tanh'


class Args_Mortality_P12_Init_LEIT_MTAN(ArgsMTAN):
    def __init__(self):
        super(Args_Mortality_P12_Init_LEIT_MTAN, self).__init__()
        self.leit_model = 'mtan'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = ''
        self.ratio_ce = 1000
        self.num_dl_workers = 8
        self.device = 'cuda:2'
        self.data = 'P12'
        self.quantization = 0.0167    # scale to 1 min interval
        self.n_samples = 8000
        self.extrap = False
        self.variable_num = 41
        self.p12_classify = True
