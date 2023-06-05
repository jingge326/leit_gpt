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


class Args_Mortality_M4_Init_LEIT_IVPVAE_CouplingFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_IVPVAE_CouplingFlow, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'r4_p5_100ce'
        self.ivp_solver = 'couplingflow'
        self.ratio_ce = 100
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:1'


class Args_Mortality_M4_Init_LEIT_IVPVAE_GRUFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_IVPVAE_GRUFlow, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'r4_p5_100ce'
        self.ivp_solver = 'gruflow'
        self.ratio_ce = 100
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:1'


class Args_Mortality_M4_Init_LEIT_ClassicGRU(ArgsIVPVAE):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_ClassicGRU, self).__init__()
        self.leit_model = 'classic_rnn'
        self.cell = 'gru'    # ['gru', 'expdecay']
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'r4_p5_5min'
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:1'


class Args_Mortality_M4_Init_LEIT_ClassicGRUD(ArgsIVPVAE):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_ClassicGRUD, self).__init__()
        self.leit_model = 'classic_rnn'
        self.cell = 'expdecay'    # ['gru', 'expdecay']
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'r4_p5'
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:0'


class Args_Mortality_M4_Init_LEIT_MTAN(ArgsMTAN):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_MTAN, self).__init__()
        self.leit_model = 'mtan'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'r4_p5'
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:0'
        self.batch_size = 16


class Args_Mortality_M4_Init_LEIT_Raindrop(ArgsRaindrop):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_Raindrop, self).__init__()
        self.leit_model = 'raindrop'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'r4_p5_5min'
        self.num_dl_workers = 8
        self.data = 'm4_mortality_500'
        self.device = 'cuda:1'
        self.batch_size = 64
        self.d_ob = 2


class Args_Mortality_M4_Init_LEIT_MTANIVP(ArgsMTANIVP):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_MTANIVP, self).__init__()
        self.leit_model = 'mtan_ivp'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        self.ivp_solver = 'resnetflow'
        self.ratio_ce = 1000
        self.test_info = 'testing'
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:2'
        self.batch_size = 64
        self.embed_time = 128


class Args_Mortality_M4_Init_LEIT_CKCONV(ArgsCKCONV):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_CKCONV, self).__init__()
        self.leit_model = 'ckconv'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'r4_p5_h60'
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:1'
        self.batch_size = 64
        self.max_len = 1440
        self.no_blocks = 2
        self.in_channels = self.variable_num * 2
        self.no_hidden = 60


class Args_Mortality_M4_Init_LEIT_REDVAE_ResNetFlow(ArgsREDVAE):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_REDVAE_ResNetFlow, self).__init__()
        self.leit_model = 'red_vae'
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
        self.ratio_ce = 100
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:1'


class Args_Mortality_M4_Init_LEIT_REDVAE_ODE(ArgsREDVAE):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_REDVAE_ODE, self).__init__()
        self.leit_model = 'red_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'testing'
        self.ivp_solver = 'ode'
        self.ratio_ce = 100
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:1'
        # For ODE
        self.activation = 'ELU'
        self.final_activation = 'Tanh'


class Args_Mortality_M4_Init_LEIT_IVPVAE_ODE_try(ArgsIVPVAE):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_IVPVAE_ODE_try, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        # self.test_info = 'nonescale_mask_mpw1_stest'
        self.test_info = 'en_ivp_pos'
        self.ivp_solver = 'ode'
        self.ratio_ce = 1000
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:1'
        # For ODE
        self.activation = 'ELU'
        self.final_activation = 'Tanh'


class Args_Mortality_M4_Init_LEIT_IVPVAE_ResNetFlow(ArgsIVPVAE):
    def __init__(self):
        super(Args_Mortality_M4_Init_LEIT_IVPVAE_ResNetFlow, self).__init__()
        self.leit_model = 'ivp_vae'
        self.task = 'biclass'
        self.model_type = 'initialize'  # ['initialize', 'reconstruct']
        self.embedding_method = 'mlp'
        self.classifier_type = 'mlp'
        self.classifier_input = 'z0'
        # ['unfreeze', 'odevae', 'embedding', 'flow', 'encoder_flow', 'decoder']
        self.freeze_opt = 'unfreeze'
        self.train_w_reconstr = True
        self.test_info = 'm_false_p1'
        self.ivp_solver = 'resnetflow'
        self.ratio_ce = 1000
        self.num_dl_workers = 8
        self.data = 'm4_mortality_100'
        self.device = 'cuda:1'
        self.train_w_mask = False
        self.mask_loss_ratio = 0.001
