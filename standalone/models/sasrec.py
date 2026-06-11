from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRecBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class SASRec(nn.Module):

    def __init__(self, num_items: int, max_seq_len: int = 50,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            SASRecBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, **kwargs):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.norm(x)
        return x

    def predict(self, seq_emb, candidate_ids=None):
        last_emb = seq_emb[:, -1, :]
        item_embs = self.item_embedding.weight[1:]
        scores = torch.matmul(last_emb, item_embs.T)
        return scores

class SASRecForFL(nn.Module):

    def __init__(self, num_items: int, max_seq_len: int = 50,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.sasrec = SASRec(num_items, max_seq_len, d_model, n_heads, n_layers, dropout)
        self.num_items = num_items

    def forward(self, input_ids, labels=None, num_neg_samples=256,
                eval_last_only=False, **kwargs):
        seq_emb = self.sasrec(input_ids)
        item_embs = self.sasrec.item_embedding.weight[1:]

        loss = None
        logits = None

        if eval_last_only and labels is None:
            last_emb = seq_emb[:, -1, :]
            logits = torch.matmul(last_emb, item_embs.T).unsqueeze(1)
        elif labels is not None and num_neg_samples > 0 and self.num_items > num_neg_samples * 2:
            mask = (labels > 0).float()
            flat_labels = (labels - 1).clamp(min=0).view(-1)
            flat_emb = seq_emb.view(-1, seq_emb.size(-1))
            flat_mask = mask.view(-1)

            pos_ids = flat_labels[flat_mask > 0]
            unique_pos = pos_ids.unique()
            neg_pool = torch.arange(self.num_items, device=input_ids.device)
            neg_ids = neg_pool[torch.randperm(self.num_items, device=input_ids.device)[:num_neg_samples]]
            sampled_ids = torch.cat([unique_pos, neg_ids]).unique()

            sampled_embs = item_embs[sampled_ids]
            sampled_logits = torch.matmul(flat_emb, sampled_embs.T)

            id_to_idx = torch.zeros(self.num_items, dtype=torch.long, device=input_ids.device)
            id_to_idx[sampled_ids] = torch.arange(len(sampled_ids), device=input_ids.device)
            remapped_labels = id_to_idx[flat_labels]

            loss = F.cross_entropy(sampled_logits, remapped_labels, reduction='none')
            loss = (loss * flat_mask).sum() / flat_mask.sum().clamp(min=1)
            logits = sampled_logits.view(seq_emb.size(0), seq_emb.size(1), -1)
        else:
            logits = torch.matmul(seq_emb, item_embs.T)
            if labels is not None:
                mask = (labels > 0).float()
                loss = F.cross_entropy(
                    logits.view(-1, self.num_items),
                    (labels - 1).clamp(min=0).view(-1),
                    reduction='none'
                )
                loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)

        return type('Output', (), {'loss': loss, 'logits': logits})()

    @property
    def config(self):
        return type('Config', (), {'hidden_size': self.sasrec.d_model})()
