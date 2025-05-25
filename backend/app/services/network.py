import torch
import torch.nn as nn
import torch.nn.functional as F
# -*- coding: utf-8 -*-

class ComplexBlock(nn.Module):
    def __init__(self, layer_size, dropout_rate=0.1):
        super(ComplexBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(layer_size, layer_size),
            nn.LayerNorm(layer_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        return out + residual

class UltraComplexFullyConnected(nn.Module):
    def __init__(self, in_features=2, out_features=3, num_blocks=6, layer_size=512, dropout_rate=0.1):
        super(UltraComplexFullyConnected, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, layer_size),
            nn.LayerNorm(layer_size),
            nn.GELU()
        )

        self.blocks = nn.ModuleList([
            ComplexBlock(layer_size, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(layer_size)
        self.output_layer = nn.Linear(layer_size, out_features)

        self._init_weights()

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = self.output_layer(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)