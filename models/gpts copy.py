import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class MultiheadAttention(nn.Module):
    # MultiheadAttention for irregular time series
    def __init__(self, args):
        super().__init__()
        self.attn_dim = args.attn_dim
        self.num_heads = args.nhead
        self.input_dim = args.variable_num * 2 + args.embed_time
        self.all_dim = self.attn_dim * self.num_heads
        self.query = nn.Linear(self.input_dim, self.all_dim)
        self.key = nn.Linear(self.input_dim, self.all_dim)
        self.value = nn.Linear(self.input_dim, self.all_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(self.all_dim, args.attn_dim)

    def forward(self, x, mask=None):
        # x: shape: [batch_size, seq_len, input_dim]
        # mask: shape: [batch_size, seq_len]
        b, n, _ = x.size()
        q = self.query(x).view(b, n, self.num_heads, self.attn_dim).transpose(
            1, 2)  # shape: [batch_size, num_heads, seq_len, input_dim]
        k = self.key(x).view(b, n, self.num_heads, self.attn_dim).transpose(
            1, 2)  # shape: [batch_size, num_heads, seq_len, input_dim]
        v = self.value(x).view(b, n, self.num_heads, self.attn_dim).transpose(
            1, 2)  # shape: [batch_size, num_heads, seq_len, input_dim]

        # Scaled Dot-Product Attention
        # shape: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(self.input_dim,
                       dtype=torch.float32, device=x.device))

        assert mask is not None
        scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2)==0, float('-inf'))

        # shape: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # shape: [batch_size, num_heads, seq_len, input_dim]
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.masked_fill(mask.unsqueeze(1).unsqueeze(3)==0, float(0))

        # Concatenation and linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            b, n, -1)  # shape: [batch_size, seq_len, num_heads*input_dim]
        # shape: [batch_size, seq_len, input_dim]
        output = self.fc(attn_output)

        return output


class TimeEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)
        utils.init_network_weights(self.periodic)
        utils.init_network_weights(self.linear)

    def forward(self, tt):
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)


class GPTS(nn.Module):
    def __init__(self, args):

        super(GPTS, self).__init__()
        self.args = args
        self.time_start = 0
        self.latent_dim = args.latent_dim
        # basic models
        self.time_embedding = TimeEmbedding(args)
        self.multi_attn = MultiheadAttention(args)

    def softmax_with_mask(self, input, mask, dim):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)
        output = torch.exp(input) * mask
        output = output / torch.sum(output, dim=dim, keepdim=True)
        return output
    
    def forward(self, batch):
        self.time_start = time.time()
        times_in = batch['times_in']
        data_in = batch['data_in']
        mask_in = batch['mask_in']
        t_exist = batch['exist_times']
        utils.check_mask(data_in, mask_in)
        time_embed = self.time_embedding(times_in.unsqueeze(-1))
        x = torch.cat((data_in, mask_in, time_embed), dim=-1)
        
        latent_states = self.multi_attn(x, mask=t_exist)

        forward_info = {'latent_states': latent_states}
        
        # return forward_info
        
        device = idx.device
        b, t = idx.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    