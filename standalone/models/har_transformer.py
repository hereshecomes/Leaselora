from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):

    def __init__(self, num_channels: int, d_model: int, patch_size: int = 16, seq_len: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(num_channels * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = x.reshape(B, self.num_patches, self.patch_size * C)
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        return x

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.ffn_up = nn.Linear(d_model, d_ff)
        self.ffn_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm1(x)

        q = self.query(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        x = x + self.dropout(out)

        h = self.norm2(x)
        ffn_out = self.ffn_down(self.dropout(F.gelu(self.ffn_up(h))))
        x = x + self.dropout(ffn_out)

        return x

class HARTransformer(nn.Module):

    def __init__(
        self,
        num_channels: int = 9,
        num_classes: int = 6,
        seq_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        patch_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(num_channels, d_model, patch_size, seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_values: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        x = self.patch_embed(input_values)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.classifier(self.dropout(cls_out))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return type('Output', (), {'loss': loss, 'logits': logits})()

HAR_TRANSFORMER_LORA_TARGETS = {
    "qv": ["query", "value"],
    "qkvo": ["query", "key", "value", "out_proj"],
    "attention_ffn": ["query", "key", "value", "out_proj", "ffn_up", "ffn_down"],
}

HAR_TRANSFORMER_DEFAULT_TARGETS = ["query", "value"]
