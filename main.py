from pathlib import Path
import traceback
import argparse

from experiments.exp_pretrain import Exp_Pretrain
from experiments.exp_biclass import Exp_BiClass
from experiments.exp_extrap import Exp_Extrap
from experiments.exp_interp import Exp_Interp


parser = argparse.ArgumentParser(
    description="Run all experiments within the Leit framework.")

# Args for the training process
parser.add_argument("--random-state", type=int, default=5, help="Random seed")
parser.add_argument("--proj-path", type=str,
                    default=str(Path(__file__).parents[0]))
parser.add_argument("--test-info", default="test")
parser.add_argument("--leit-model", default="gpts",
                    choices=["gpts", "ivp_vae", "ivp_vae_old", "red_vae", "classic_rnn", "mtan",
                             "raindrop", "ckconv", "cru", "gob", "grud",
                             "ivp_auto", "att_ivp_vae", "ivp_auto"])
parser.add_argument("--model-type", default="initialize",
                    choices=["initialize", "reconstruct"])
parser.add_argument("--num-dl-workers", type=int, default=32)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--epochs-min", type=int, default=1)
parser.add_argument("--epochs-max", type=int, default=1000,
                    help="Max training epochs")
parser.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience")
parser.add_argument("--weight-decay", type=float,
                    default=0.0001, help="Weight decay (regularization)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--lr-scheduler-step", type=int, default=40,
                    help="Every how many steps to perform lr decay")
parser.add_argument("--lr-decay", type=float, default=0.5,
                    help="Multiplicative lr decay factor")
parser.add_argument("--clip-gradient", action='store_false')
parser.add_argument("--clip", type=float, default=1)
parser.add_argument("--not-vae", action='store_false')
parser.add_argument("--para-file-type", default="pl_ckpt",
                    choices=["pl_ckpt", "pt"])
parser.add_argument("--freeze-opt", default="unfreeze",
                    choices=["unfreeze", "odevae", "embedding", "flow", "encoder_flow", "decoder"])
parser.add_argument("--num-class", type=int, default=11)
parser.add_argument("--z0_mapper", default="softplus",
                    choices=["logstd", "softplus"])
parser.add_argument("--log-tool", default="wandb",
                    choices=["logging", "wandb", "all"])

# Args for datasets
parser.add_argument("--data", default="m4_gpts", help="Dataset name",
                    choices=["m4_gpts", "m4_general", "p12", "p19_sepsis", "person_activity",
                             "m4_mortality_100", "m4_mortality_250", "m4_mortality_500", "m4_mortality_1000",
                             "m4_mortality_2000", "m4_mortality_3000", "m4_next", "m4_next_100", "m4_next_250",
                             "m4_next_500", "m4_next_1000", "m4_next_2000", "m4_next_3000", "eicu", "m4",
                             "m4_smooth"])
parser.add_argument("--num-samples", type=int, default=-1)
parser.add_argument("--variable-num", type=int,
                    default=113, choices=[96, 41, 34, 14, 113])
parser.add_argument("--ts-full", action='store_true')
parser.add_argument("--del-std5", action='store_true')
parser.add_argument("--time-scale", default="time_max",
                    choices=["time_max", "self_max", "constant", "none", "max"])
parser.add_argument("--time-constant", type=float, default=1)
parser.add_argument("--first-dim", default="batch",
                    choices=["batch", "time_series"])
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--t-offset", type=float, default=0)
parser.add_argument("--ml-task", default="pretrain",
                    choices=["biclass", "extrap", "interp", "length", "pretrain"])
parser.add_argument("--extrap-full", action='store_true')
parser.add_argument("--p12-classify", action='store_false')
parser.add_argument("--down-times", type=int, default=1,
                    help="downsampling timestamps")
parser.add_argument("--time-len-fixed", action='store_true')
parser.add_argument("--time-max", type=int, default=1439)
parser.add_argument("--num-times", type=int, default=1440)
parser.add_argument("--next-start", type=float, default=1440)
parser.add_argument("--next-end", type=float, default=2880)
parser.add_argument("--next-headn", type=int, default=0)
parser.add_argument('--sample-tp', type=float, default=1.0)
parser.add_argument('--times-drop', type=float, default=0.0)
parser.add_argument('--mask-drop-rate', type=float, default=0.0)

# Args for IVP solvers
parser.add_argument("--ivp-solver", default="resnetflow",
                    choices=["resnetflow", "couplingflow", "gruflow", "ode"])
parser.add_argument("--flow-layers", type=int, default=2,
                    help="Number of flow layers")
parser.add_argument("--hidden-layers", type=int, default=3,
                    help="Number of hidden layers")
parser.add_argument("--hidden-dim", type=int, default=128,
                    help="Size of hidden layer")
parser.add_argument("--activation", type=str, default="ELU",
                    help="Hidden layer activation")
parser.add_argument("--final-activation", type=str,
                    default="Tanh", help="Last layer activation")
parser.add_argument("--odenet", type=str, default="concat",
                    help="Type of ODE network", choices=["concat", "gru"])  # gru only in GOB
parser.add_argument("--ode_solver", type=str, default="dopri5",
                    help="ODE solver", choices=["dopri5", "rk4", "euler"])
parser.add_argument("--solver_step", type=float,
                    default=0.05, help="Fixed solver step")
parser.add_argument("--atol", type=float, default=1e-4,
                    help="Absolute tolerance")
parser.add_argument("--rtol", type=float, default=1e-3,
                    help="Relative tolerance")
parser.add_argument("--time-net", type=str, default="TimeTanh", help="Name of time net",
                    choices=["TimeFourier", "TimeFourierBounded", "TimeLinear", "TimeTanh"])
parser.add_argument("--time-hidden-dim", type=int, default=8,
                    help="Number of time features (only for Fourier)")

# Args for VAE
parser.add_argument("--k-iwae", type=int, default=3)
parser.add_argument("--kl-coef", type=float, default=1.0)
parser.add_argument("--latent-dim", type=int, default=20)
parser.add_argument("--attn-dim", type=int, default=20)
parser.add_argument("--encoder-dim", type=int, default=20)
parser.add_argument("--classifier-input", default="z0")
parser.add_argument("--train-w-reconstr", action='store_false')
parser.add_argument("--ratio-ce", type=float, default=1000)
parser.add_argument("--ratio-nll", type=float, default=1)
parser.add_argument("--ratio-ae", type=float, default=1)
parser.add_argument("--ratio-zz", type=float, default=0)
parser.add_argument("--prior-mu", type=float, default=0.0)
parser.add_argument("--prior-std", type=float, default=1.0)
parser.add_argument("--obsrv-std", type=float, default=0.01)
parser.add_argument("--draw-all-times", action='store_true')
parser.add_argument("--fast-llloss", action='store_true')

# Args for RNN
parser.add_argument("--rnn-hidden-dim", type=int, default=20)
parser.add_argument("--rnn-cell", default="gru", choices=["gru"])
parser.add_argument("--run-backwards", action='store_false')
parser.add_argument("--input-space-decay", action='store_true')
parser.add_argument("--embedding-method", default="mlp",
                    choices=["mlp", "linear"])
parser.add_argument("--reconstr-method", default="mlp",
                    choices=["mlp", "linear"])
parser.add_argument("--extrap-method", default="mlp",
                    choices=["mlp", "seq2seq"])

# Args for IVPVAE
parser.add_argument("--test-relu", action='store_false')
parser.add_argument("--train-w-mask", action='store_true')
parser.add_argument("--mask-loss-ratio", type=float, default=0.01,
                    choices=[1, 0.1, 0.01, 0.001])
parser.add_argument("--combine-methods", default="average",
                    choices=["average", "attn_latent", "attn_rough", "attn_embed", "attn_init", "kl_weighted"])
parser.add_argument("--decoupled-ae", action='store_true')

# Args for mTAN
parser.add_argument("--rec-hidden", type=int, default=256)
parser.add_argument("--gen-hidden", type=int, default=50)
parser.add_argument("--embed-time", type=int, default=128)
parser.add_argument("--enc", default="mtan_rnn")
parser.add_argument("--dec", default="mtan_rnn")
parser.add_argument("--split", default=0)
parser.add_argument("--n-samples", type=int, default=8000)
parser.add_argument("--norm", action='store_false')
parser.add_argument("--kl", action='store_false')
parser.add_argument("--learn-emb", action='store_false')
parser.add_argument("--old_split", action='store_false')
parser.add_argument("--enc-num-heads", type=int, default=1)
parser.add_argument("--dec-num-heads", type=int, default=1)
parser.add_argument("--num-ref-points", type=int, default=128)
parser.add_argument("--classify-pertp", action='store_true')

# Args for Raindrop
parser.add_argument("--d-ob", type=int, default=4, help="")
parser.add_argument("--dim-pos-encoder", type=int, default=16)
parser.add_argument("--nhead", type=int, default=12,
                    help="number of heads in multihead-attention")
parser.add_argument("--nhid", type=int, default=128,
                    help="Dimension of feedforward network model")
parser.add_argument("--nlayers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Specifies a layer-wise dropout factor")
parser.add_argument("--MAX", default=100,
                    help="positional encoder MAX parameter")
parser.add_argument("--perc", type=float, default=0.5)
parser.add_argument("--aggreg", default="mean")
parser.add_argument("--n-classes", type=int, default=2,
                    help="number of classes")
parser.add_argument("--global-structure", default=None)
parser.add_argument("--classifier-type", default="mlp")
parser.add_argument("--sensor-wise-mask", action='store_true')

# Args for CKCONV
parser.add_argument("--no-hidden", type=int, default=30,
                    help="The number of channels at the hidden layers of the main network")
parser.add_argument("--no-blocks", type=int, default=2,
                    help="The number of residual blocks in the network")
parser.add_argument("--weight-dropout", type=float, default=0.0,
                    help="The number of residual blocks in the network")
parser.add_argument("--kernelnet-norm-type", default="LayerNorm",
                    help="""If model == CKCNN, the normalization 
                    type to be used in the MLPs parameterizing the 
                    convolutional kernels. If kernelnet-activation-function==Sine,
                    no normalization will be used. e.g., LayerNorm.""")
parser.add_argument("--kernelnet-activation-function", default="Sine",
                    help="The activation function used in the MLPs parameterizing the convolutional")
parser.add_argument("--kernelnet-no-hidden", type=int, default=32,
                    help="The number of hidden units used in the MLPs parameterizing the convolutional kernels")
parser.add_argument("--kernelnet-omega-0", default=9.779406396796968)
parser.add_argument("--dim-linear", type=int, default=1)
parser.add_argument("--in-channels", type=int, default=75)
parser.add_argument("--mask-type", default="cumsum")

# Args for CRU
parser.add_argument('--cru-hidden-units', type=int, default=50,
                    help="Hidden units of encoder and decoder.")
parser.add_argument('--num-basis', type=int, default=20,
                    help="Number of basis matrices to use in transition model for locally-linear transitions. K in paper")
parser.add_argument('--bandwidth', type=int, default=10,
                    help="Bandwidth for basis matrices A_k. b in paper")
parser.add_argument('--enc-var-activation', type=str, default='square',
                    help="Variance activation function in encoder. Possible values elup1, exp, relu, square")
parser.add_argument('--dec-var-activation', type=str, default='exp',
                    help="Variance activation function in decoder. Possible values elup1, exp, relu, square")
parser.add_argument('--trans-net-hidden-activation', type=str,
                    default='tanh', help="Activation function for transition net.")
parser.add_argument('--trans-net-hidden-units', type=list,
                    default=[], help="Hidden units of transition net.")
parser.add_argument('--trans-var-activation', type=str,
                    default='relu', help="Activation function for transition net.")
parser.add_argument('--learn-trans-covar', type=bool,
                    default=True, help="If to learn transition covariance.")
parser.add_argument('--learn-initial-state-covar', action='store_true',
                    help="If to learn the initial state covariance.")
parser.add_argument('--initial-state-covar', type=int,
                    default=1, help="Value of initial state covariance.")
parser.add_argument('--trans-covar', type=float, default=0.1,
                    help="Value of initial transition covariance.")
parser.add_argument('--t-sensitive-trans-net',  action='store_true',
                    help="If to provide the time gap as additional input to the transition net. Used for RKN-Delta_t model in paper")
parser.add_argument('--f-cru',  action='store_true',
                    help="If to use fast transitions based on eigendecomposition of the state transitions (f-CRU variant).")
parser.add_argument('--rkn',  action='store_true',
                    help="If to use discrete state transitions (RKN baseline).")
parser.add_argument('--orthogonal', type=bool, default=True,
                    help="If to use orthogonal basis matrices in the f-CRU variant.")
parser.add_argument('--extrap_w', type=float, default=0)

# GRU-ODE-Bayes specific args
parser.add_argument('--mixing', type=float, default=0.0001,
                    help='Ratio between KL and update loss')
parser.add_argument('--gob_prep_hidden', type=int, default=10,
                    help='Size of hidden state for covariates')
parser.add_argument('--gob_cov_hidden', type=int, default=50,
                    help='Size of hidden state for covariates')
parser.add_argument('--gob_p_hidden', type=int, default=25,
                    help='Size of hidden state for initialization')
parser.add_argument('--invertible', type=int, default=1,
                    help='If network is invertible', choices=[0, 1])

# GPTS specific args
parser.add_argument("--mhatt_n_layer", type=int, default=1)
parser.add_argument("--n_embd", type=int, default=768)
parser.add_argument("--seq_len_min", type=int, default=1)
parser.add_argument("--seq_len_max", type=int, default=256)
parser.add_argument("--bias", action='store_true')
parser.add_argument("--gpts_output", default="last", choices=["all", "last"])
parser.add_argument(
    "--pre_model", default="gpts_pretrain_initialize_nhead12_nlyrs4_bsize64_min2_r1.pt")
parser.add_argument("--evolve_module", default="delta_t",
                    choices=["ivp", "delta_t"])

parser.add_argument("--last_ivp", action='store_true')
parser.add_argument("--use_auxiliary_loss", action='store_true')
parser.add_argument("--del_bad_p12", action='store_true')
parser.add_argument("--train_obj", default="gpt", choices=["gpt", "bert"])
parser.add_argument("--add_cls", action='store_true')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.ml_task == 'extrap':
        experiment = Exp_Extrap(args)
    elif args.ml_task == 'interp':
        experiment = Exp_Interp(args)
    elif args.ml_task == 'biclass':
        experiment = Exp_BiClass(args)
    elif args.ml_task == 'pretrain':
        experiment = Exp_Pretrain(args)
    else:
        raise ValueError("Unknown")

    # try:
    #     experiment.run()
    #     experiment.finish()
    # except Exception:
    #     with open(experiment.proj_path/"log"/"err_{}.log".format(experiment.args.exp_name), "w") as fout:
    #         print(traceback.format_exc(), file=fout)

    experiment.run()
    experiment.finish()
