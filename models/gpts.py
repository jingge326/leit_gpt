import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.ivp_solvers import CouplingFlow, ODEModel, ResNetFlow, GRUFlow
import utils


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.nhead == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.nhead = config.nhead
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.nhead, C //
                   self.nhead).transpose(1, 2)  # (B, nh, T, hs)

        mask_left_attn = torch.tril(torch.ones(
            T, T)).view(1, 1, T, T).to(x.device)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask_left_attn == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)
        utils.init_network_weights(self.periodic)
        utils.init_network_weights(self.linear)

    def forward(self, tt):
        tt = torch.log(tt + 1e-5)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)


def build_ivp_solver(args):
    ivp_solver = None
    hidden_dims = [args.hidden_dim] * args.hidden_layers
    if args.ivp_solver == 'ode':
        ivp_solver = utils.SolverWrapper(ODEModel(args.n_embd, args.odenet, hidden_dims, args.activation,
                                                  args.final_activation, args.ode_solver, args.solver_step, args.atol, args.rtol))
    else:
        if args.ivp_solver == 'couplingflow':
            flow = CouplingFlow
        elif args.ivp_solver == 'resnetflow':
            flow = ResNetFlow
        elif args.ivp_solver == 'gruflow':
            flow = GRUFlow
        else:
            raise NotImplementedError

        ivp_solver = utils.SolverWrapper(flow(
            args.n_embd//args.nhead, args.flow_layers, hidden_dims, args.time_net, args.time_hidden_dim))
    return ivp_solver


class StatesMapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mapper = torch.nn.Sequential(nn.Linear(args.n_embd + 1, args.n_embd + 1),
                                          nn.ReLU(),
                                          nn.Linear(args.n_embd + 1, args.n_embd))

    def forward(self, x, t, backwards=False):
        assert len(x.shape) - len(t.shape) == 1
        t = t.unsqueeze(-1)
        if t.shape[-3] == 1:
            t = t.repeat_interleave(x.shape[-3], dim=-3)
        if len(x.shape) == 4 and x.shape[0] != t.shape[0]:
            t = t.repeat_interleave(x.shape[0], dim=0)
        x = torch.cat([x, t], dim=-1)
        y = self.mapper(x)
        return y


class GPTS(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.register_buffer('zero_tensor', torch.zeros([1]))
        self.input_lyr = nn.Linear(
            args.variable_num * 2 + args.embed_time, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)
        self.multi_attn_lyrs = nn.Sequential(
            *[Block(args) for _ in range(args.mhatt_n_layer)])
        self.ln_f = LayerNorm(args.n_embd, bias=args.bias)
        self.lm_head = nn.Linear(args.n_embd, args.variable_num)
        self.time_embedding = TimeEmbedding(args)

        if args.evolve_module == "ivp":
            self.evolve = build_ivp_solver(args)
        elif args.evolve_module == "delta_t":
            self.evolve = StatesMapper(args)
        else:
            raise NotImplementedError

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * args.mhatt_n_layer))

        # report number of parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (self.num_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, batch):
        results = {}
        self.time_start = time.time()
        times_in = batch['times_in']
        data_in = batch['data_in']
        mask_in = batch['mask_in']
        utils.check_mask(data_in, mask_in)
        time_embed = self.time_embedding(times_in.unsqueeze(-1))
        x = torch.cat((data_in, mask_in, time_embed), dim=-1)
        x = self.input_lyr(x)

        x = self.multi_attn_lyrs(x)

        x = self.ln_f(x)

        results["latent_states"] = x

        return results

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, nhead and n_embd are determined from model_type
        config_args = {
            # 124M params
            'gpt2':         dict(n_layer=12, nhead=12, n_embd=768),
            # 350M params
            'gpt2-medium':  dict(n_layer=24, nhead=16, n_embd=1024),
            # 774M params
            'gpt2-large':   dict(n_layer=36, nhead=20, n_embd=1280),
            # 1558M params
            'gpt2-xl':      dict(n_layer=48, nhead=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        # always 50257 for GPT model checkpoints
        config_args['vocab_size'] = 50257
        # always 1024 for GPT model checkpoints
        config_args['block_size'] = 1024
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.nhead, cfg.n_embd//cfg.nhead, cfg.block_size
        flops_per_token = 6*self.num_params + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def generate_seq(self, previous_state, delta_ts):
        n_steps = mask.size(1)
        outputs = []
        x = first_input
        for i in range(n_steps):
            delta_t = delta_ts[:, i:i+1]

            time_embed = self.time_embedding(times_in.unsqueeze(-1))
            x = torch.cat((data_in, mask_in, time_embed), dim=-1)
            x = self.input_lyr(x)
            x = self.multi_attn_lyrs(x)
            x = self.ln_f(x)

            results["latent_states"] = x

            if mask is not None:
                mask_i = mask[:, i, :]
                x = torch.cat((x, mask_i), -1)

            input_w_t = torch.cat((x, delta_t), -1).squeeze(1)
            hidden = cell(input_w_t, hidden)
            rnn_output = decoder(hidden)
            outputs.append(rnn_output)
            x = rnn_output

        outputs = torch.stack(outputs, 0).permute(1, 0, 2).unsqueeze(0)
        return outputs
