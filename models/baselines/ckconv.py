import time
import torch
from libs.ckconv import nn


class CKCONV(torch.nn.Module):
    def __init__(self, args):
        super(CKCONV, self).__init__()
        self.time_start = 0
        in_channels = args.in_channels
        hidden_channels = args.no_hidden
        num_blocks = args.no_blocks
        kernelnet_hidden_channels = args.kernelnet_no_hidden
        kernelnet_activation_function = args.kernelnet_activation_function
        kernelnet_norm_type = args.kernelnet_norm_type
        dim_linear = args.dim_linear
        bias = True
        omega_0 = args.kernelnet_omega_0
        dropout = args.dropout
        weight_dropout = args.weight_dropout
        pool = False  # Always False in our experiments.
        out_channels = None

        blocks = []
        for i in range(num_blocks):
            block_in_channels = in_channels if i == 0 else hidden_channels
            block_out_channels = hidden_channels
            if i == num_blocks-1 and out_channels is not None:
                block_out_channels = out_channels
            blocks.append(
                nn.CKBlock(
                    block_in_channels,
                    block_out_channels,
                    kernelnet_hidden_channels,
                    kernelnet_activation_function,
                    kernelnet_norm_type,
                    dim_linear,
                    bias,
                    omega_0,
                    dropout,
                    weight_dropout,
                )
            )
            if pool:
                blocks.append(torch.nn.MaxPool1d(kernel_size=2))
        self.backbone = torch.nn.Sequential(*blocks)

    def forward(self, x):
        self.time_start = time.time()
        return self.backbone(x)
