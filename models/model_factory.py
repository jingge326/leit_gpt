from base64 import decode
from collections import OrderedDict
import torch
from models.base_models.mtan_components import dec_mtan_rnn, enc_mtan_ivp, enc_mtan_rnn
from models.baselines.classic_rnn import ClassicRNN
from models.baselines.mtan import MTAN
from models.baselines.mtan_ivp import MTANIVP

from models.ivp_solvers import CouplingFlow, ODEModel, ResNetFlow, GRUFlow
from models.base_models.ivpvae_components import Embedding_Linear, Embedding_MLP, Reconst_DM_Mapper_ReLU, BinaryClassifier, MLP_REGRE, Embedding_MLP, Reconst_Mapper_Linear, Reconst_Mapper_MLP
from models.base_models.embedding import Embedding_GNN
from models.models_interp import CRU_Interp, ClassicRNN_Interp, IVPVAE_Interp, MTAN_Interp, REDVAE_Interp
from models.models_multiclass import CKCONV_MultiClass, ClassicRNN_MultiClass, IVPVAE_MultiClass, MTAN_MultiClass, MTANIVP_MultiClass, REDVAE_MultiClass, Raindrop_MultiClass
from models.pl_wrapper import PL_Wrapper

from utils import SolverWrapper
from models.ivp_vae import IVPVAE
from models.models_extrap import AttIVPVAE_Extrap, CKCONV_Extrap, CRU_Extrap, ClassicRNN_Extrap, GOB_Extrap, GPTS_Extrap, GRUD_Extrap, IVPAuto_Extrap, IVPAuto_Extrap, IVPVAE_Extrap, IVPVAE_OLD_Extrap, MTAN_Extrap, REDVAE_Extrap
from models.models_biclass import AttIVPVAE_BiClass, ClassicRNN_BiClass, GPTS_BiClass, GRUD_BiClass, IVPAuto_BiClass, IVPVAE_OLD_BiClass, MTAN_BiClass, MTANIVP_BiClass, IVPVAE_BiClass, REDVAE_BiClass, Raindrop_BiClass, CKCONV_BiClass
from models.models_length import Leit_Length


class ModelFactory:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def build_ivp_solver(self, states_dim):
        ivp_solver = None
        hidden_dims = [self.args.hidden_dim] * self.args.hidden_layers
        if self.args.ivp_solver == 'ode':
            ivp_solver = SolverWrapper(ODEModel(states_dim, self.args.odenet, hidden_dims, self.args.activation,
                                                self.args.final_activation, self.args.ode_solver, self.args.solver_step, self.args.atol, self.args.rtol))
        else:
            if self.args.ivp_solver == 'couplingflow':
                flow = CouplingFlow
            elif self.args.ivp_solver == 'resnetflow':
                flow = ResNetFlow
            elif self.args.ivp_solver == 'gruflow':
                flow = GRUFlow
            else:
                raise NotImplementedError

            ivp_solver = SolverWrapper(flow(
                states_dim, self.args.flow_layers, hidden_dims, self.args.time_net, self.args.time_hidden_dim))
        return ivp_solver

    def init_ivpvae_components(self):

        if self.args.embedding_method == "mlp":
            embedding_nn = Embedding_MLP(
                self.args.variable_num, self.args.latent_dim)
        elif self.args.embedding_method == "linear":
            embedding_nn = Embedding_Linear(
                self.args.variable_num, self.args.latent_dim)
        else:
            raise ValueError("Unknown!")

        ivp_solver = self.build_ivp_solver(self.args.latent_dim)

        if self.args.train_w_mask:
            reconst_mapper = Reconst_DM_Mapper_ReLU(
                self.args.latent_dim, self.args.variable_num)
        else:
            if self.args.reconstr_method == "mlp":
                reconst_mapper = Reconst_Mapper_MLP(
                    self.args.latent_dim, self.args.variable_num)
            elif self.args.reconstr_method == "linear":
                reconst_mapper = Reconst_Mapper_Linear(
                    self.args.latent_dim, self.args.variable_num)
            else:
                raise ValueError("Unknown!")

        return embedding_nn, ivp_solver, reconst_mapper

    def _switch_embedding_nn(self, model):
        model.embedding_nn = Embedding_MLP(
            self.args.variable_num, self.args.latent_dim)
        return model

    def _switch_reconst_mapper(self, model):
        model.reconst_mapper = Reconst_Mapper_MLP(
            self.args.latent_dim, self.args.variable_num)
        return model

    def _append_classifier(self, model):
        model.classifier = BinaryClassifier(self.args.latent_dim)
        return model

    def _append_regre_model(self, model):
        model.regre_model = MLP_REGRE(self.args.latent_dim)
        return model

    def _modify_state_dict_keys(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('core_model.'):
                k = k[11:]
            new_state_dict[k] = v
        return new_state_dict

    def load_pl_pretrain_model(self, load_para_path):
        return PL_Wrapper(ClassicRNN(args=self.args), self.args).load_from_checkpoint(load_para_path)

    def reconstruct_pl_length_model(self, model, load_para_path):
        state_dict = self._modify_state_dict_keys(
            torch.load(load_para_path)['state_dict'])
        model.load_state_dict(state_dict, strict=False)
        return model

    def reconstruct_pl_biclass_model(self, model, load_para_path):
        state_dict = self._modify_state_dict_keys(
            torch.load(load_para_path)['state_dict'])
        model.load_state_dict(state_dict, strict=True)
        return model

    def reconstruct_models(self, model, load_para_path, strict=False):
        loaded_state_dict = torch.load(
            load_para_path, map_location=self.args.device)

        if strict == False:
            del_keys = ["input_lyr.bias", "input_lyr.weight",
                        "lm_head.weight", "lm_head.bias"]
            for key in del_keys:
                del loaded_state_dict[key]

        keys_bad = model.load_state_dict(loaded_state_dict, strict)

        # Print the missing and unexpected keys
        self.logger.info("Missing keys:")
        for key in keys_bad.missing_keys:
            self.logger.info(key)

        self.logger.info("Unexpected keys:")
        for key in keys_bad.unexpected_keys:
            self.logger.info(key)

        return model

    def initialize_biclass_model(self):

        if self.args.leit_model == 'ivp_vae':
            embedding_nn, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

            return IVPVAE_BiClass(
                args=self.args,
                embedding_nn=embedding_nn,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'gpts':
            return GPTS_BiClass(args=self.args)

        elif self.args.leit_model == 'ivp_vae_old':
            embedding_nn, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

            return IVPVAE_OLD_BiClass(
                args=self.args,
                embedding_nn=embedding_nn,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'att_ivp_vae':
            embedding_nn, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

            return AttIVPVAE_BiClass(
                args=self.args,
                embedding_nn=embedding_nn,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'ivp_auto':
            embedding_nn, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

            return IVPAuto_BiClass(
                args=self.args,
                embedding_nn=embedding_nn,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'red_vae':
            z0_diffeq_solver = self.build_ivp_solver(self.args.encoder_dim)
            diffeq_solver = self.build_ivp_solver(self.args.latent_dim)
            return REDVAE_BiClass(
                args=self.args,
                z0_diffeq_solver=z0_diffeq_solver,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'classic_rnn':
            return ClassicRNN_BiClass(args=self.args)

        elif self.args.leit_model == 'grud':
            return GRUD_BiClass(args=self.args)

        elif self.args.leit_model == 'mtan':
            encoder_z0 = enc_mtan_rnn(
                input_dim=self.args.variable_num,
                latent_dim=self.args.latent_dim,
                num_ref_points=self.args.num_ref_points,
                nhidden=self.args.rec_hidden,
                embed_time=128,
                learn_emb=self.args.learn_emb,
                num_heads=self.args.enc_num_heads)
            decoder = dec_mtan_rnn(
                input_dim=self.args.variable_num,
                latent_dim=self.args.latent_dim,
                num_ref_points=self.args.num_ref_points,
                nhidden=self.args.gen_hidden,
                embed_time=128,
                learn_emb=self.args.learn_emb,
                num_heads=self.args.dec_num_heads)
            return MTAN_BiClass(args=self.args, encoder_z0=encoder_z0, decoder=decoder)

        elif self.args.leit_model == 'mtan_ivp':
            encoder_z0 = enc_mtan_ivp(self.args)
            diffeq_solver = self.build_ivp_solver(self.args.latent_dim)
            reconst_mapper = Reconst_Mapper_MLP(
                self.args.latent_dim, self.args.variable_num)

            return MTANIVP_BiClass(
                args=self.args,
                encoder_z0=encoder_z0,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'raindrop':
            return Raindrop_BiClass(args=self.args)

        elif self.args.leit_model == 'ckconv':
            return CKCONV_BiClass(args=self.args)

        else:
            raise ValueError('Wrong LEIT model!')

    def initialize_multiclass_model(self):

        if self.args.leit_model == 'ivp_vae':
            if self.args.test_relu:
                embedding_nn, encoder_z0, diffeq_solver, reconst_mapper = self.init_ivpvae_components()
            else:
                embedding_nn, encoder_z0, diffeq_solver, reconst_mapper = self.init_ivpvae_components()
            return IVPVAE_MultiClass(
                args=self.args,
                embedding_nn=embedding_nn,
                encoder_z0=encoder_z0,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'red_vae':
            z0_diffeq_solver = self.build_ivp_solver(self.args.encoder_dim)
            diffeq_solver = self.build_ivp_solver(self.args.latent_dim)
            return REDVAE_MultiClass(
                args=self.args,
                z0_diffeq_solver=z0_diffeq_solver,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'classic_rnn':
            return ClassicRNN_MultiClass(args=self.args)

        elif self.args.leit_model == 'mtan':
            encoder_z0 = enc_mtan_rnn(
                input_dim=self.args.variable_num,
                latent_dim=self.args.latent_dim,
                num_ref_points=self.args.num_ref_points,
                nhidden=self.args.rec_hidden,
                embed_time=128,
                learn_emb=self.args.learn_emb,
                num_heads=self.args.enc_num_heads)
            decoder = dec_mtan_rnn(
                input_dim=self.args.variable_num,
                latent_dim=self.args.latent_dim,
                num_ref_points=self.args.num_ref_points,
                nhidden=self.args.gen_hidden,
                embed_time=128,
                learn_emb=self.args.learn_emb,
                num_heads=self.args.dec_num_heads)
            return MTAN_MultiClass(args=self.args, encoder_z0=encoder_z0, decoder=decoder)

        elif self.args.leit_model == 'mtan_ivp':
            encoder_z0 = enc_mtan_ivp(self.args)
            diffeq_solver = self.build_ivp_solver(self.args.latent_dim)
            reconst_mapper = Reconst_Mapper_MLP(
                self.args.latent_dim, self.args.variable_num)

            return MTANIVP_MultiClass(
                args=self.args,
                encoder_z0=encoder_z0,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'raindrop':
            return Raindrop_MultiClass(args=self.args)

        elif self.args.leit_model == 'ckconv':
            return CKCONV_MultiClass(args=self.args)

        else:
            raise ValueError('Wrong LEIT model!')

    def initialize_length_model(self):

        embedding_nn, encoder_z0, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

        return Leit_Length(
            input_dim=self.args.variable_num,
            latent_dim=self.args.latent_dim,
            embedding_nn=embedding_nn,
            encoder_z0=encoder_z0,
            reconst_mapper=reconst_mapper,
            diffeq_solver=diffeq_solver,
            reg_input=self.args.reg_input
        )

    def initialize_pretrain_model(self):
        if self.args.leit_model == 'ivp_vae':
            embedding_nn, encoder_z0, diffeq_solver, reconst_mapper = self.init_ivpvae_components()
            return PL_Wrapper(IVPVAE(
                args=self.args,
                embedding_nn=embedding_nn,
                encoder_z0=encoder_z0,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver), self.args)

        elif self.args.leit_model == 'classic_rnn':
            return PL_Wrapper(ClassicRNN(args=self.args), self.args)

        elif self.args.leit_model == 'mtan':
            encoder_z0, decoder = self._init_mtan_components()
            return PL_Wrapper(MTAN(
                args=self.args,
                encoder_z0=encoder_z0,
                decoder=decoder), self.args)

        elif self.args.leit_model == 'mtan_ivp':
            encoder_z0, diffeq_solver, reconst_mapper = self._init_mtanivp_components()
            return PL_Wrapper(MTANIVP(
                args=self.args,
                encoder_z0=encoder_z0,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver), self.args)

        else:
            raise ValueError('Unknown leit model!')

    def initialize_extrap_model(self):

        if self.args.leit_model == 'ivp_vae':
            embedding_nn, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

            return IVPVAE_Extrap(
                args=self.args,
                embedding_nn=embedding_nn,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'ivp_vae_old':
            embedding_nn, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

            return IVPVAE_OLD_Extrap(
                args=self.args,
                embedding_nn=embedding_nn,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'ivp_auto':
            embedding_nn, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

            return IVPAuto_Extrap(
                args=self.args,
                embedding_nn=embedding_nn,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'att_ivp_vae':
            embedding_nn, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

            return AttIVPVAE_Extrap(
                args=self.args,
                embedding_nn=embedding_nn,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'red_vae':
            z0_diffeq_solver = self.build_ivp_solver(self.args.encoder_dim)
            diffeq_solver = self.build_ivp_solver(self.args.latent_dim)
            return REDVAE_Extrap(
                args=self.args,
                z0_diffeq_solver=z0_diffeq_solver,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'classic_rnn':
            return ClassicRNN_Extrap(args=self.args)

        elif self.args.leit_model == 'grud':
            return GRUD_Extrap(args=self.args)

        elif self.args.leit_model == 'mtan':
            encoder_z0 = enc_mtan_rnn(
                input_dim=self.args.variable_num,
                latent_dim=self.args.latent_dim,
                num_ref_points=self.args.num_ref_points,
                nhidden=self.args.rec_hidden,
                embed_time=128,
                learn_emb=self.args.learn_emb,
                num_heads=self.args.enc_num_heads)
            decoder = dec_mtan_rnn(
                input_dim=self.args.variable_num,
                latent_dim=self.args.latent_dim,
                num_ref_points=self.args.num_ref_points,
                nhidden=self.args.gen_hidden,
                embed_time=128,
                learn_emb=self.args.learn_emb,
                num_heads=self.args.dec_num_heads)
            return MTAN_Extrap(args=self.args, encoder_z0=encoder_z0, decoder=decoder)

        elif self.args.leit_model == 'cru':
            return CRU_Extrap(args=self.args)

        elif self.args.leit_model == 'ckconv':
            return CKCONV_Extrap(args=self.args)

        elif self.args.leit_model == 'gob':
            return GOB_Extrap(args=self.args)

        elif self.args.leit_model == 'gpts':
            return GPTS_Extrap(args=self.args)

        else:
            raise ValueError('Wrong LEIT model!')

    def initialize_interp_model(self):

        if self.args.leit_model == 'ivp_vae':
            embedding_nn, diffeq_solver, reconst_mapper = self.init_ivpvae_components()

            return IVPVAE_Interp(
                args=self.args,
                embedding_nn=embedding_nn,
                reconst_mapper=reconst_mapper,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'red_vae':
            z0_diffeq_solver = self.build_ivp_solver(self.args.encoder_dim)
            diffeq_solver = self.build_ivp_solver(self.args.latent_dim)
            return REDVAE_Interp(
                args=self.args,
                z0_diffeq_solver=z0_diffeq_solver,
                diffeq_solver=diffeq_solver)

        elif self.args.leit_model == 'classic_rnn':
            return ClassicRNN_Interp(args=self.args)

        elif self.args.leit_model == 'mtan':
            encoder_z0 = enc_mtan_rnn(
                input_dim=self.args.variable_num,
                latent_dim=self.args.latent_dim,
                num_ref_points=self.args.num_ref_points,
                nhidden=self.args.rec_hidden,
                embed_time=128,
                learn_emb=self.args.learn_emb,
                num_heads=self.args.enc_num_heads)
            decoder = dec_mtan_rnn(
                input_dim=self.args.variable_num,
                latent_dim=self.args.latent_dim,
                num_ref_points=self.args.num_ref_points,
                nhidden=self.args.gen_hidden,
                embed_time=128,
                learn_emb=self.args.learn_emb,
                num_heads=self.args.dec_num_heads)
            return MTAN_Interp(args=self.args, encoder_z0=encoder_z0, decoder=decoder)

        elif self.args.leit_model == 'cru':
            return CRU_Interp(args=self.args)

        else:
            raise ValueError('Wrong LEIT model!')
