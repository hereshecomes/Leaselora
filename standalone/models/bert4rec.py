from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT4RecBlock(nn.Module):
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

class BERT4Rec(nn.Module):

    def __init__(self, num_items: int, max_seq_len: int = 50,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 2,
                 dropout: float = 0.1, mask_prob: float = 0.2):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1

        self.item_embedding = nn.Embedding(num_items + 2, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            BERT4RecBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, **kwargs):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def predict(self, seq_emb, candidate_ids=None):
        last_emb = seq_emb[:, -1, :]
        item_embs = self.item_embedding.weight[1:self.num_items + 1]
        scores = torch.matmul(last_emb, item_embs.T)
        return scores

class BERT4RecForFL(nn.Module):

    def __init__(self, num_items: int, max_seq_len: int = 50,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 2,
                 dropout: float = 0.1, mask_prob: float = 0.2):
        super().__init__()
        self.bert4rec = BERT4Rec(
            num_items, max_seq_len, d_model, n_heads, n_layers, dropout, mask_prob)
        self.num_items = num_items
        self.mask_prob = mask_prob

    def forward(self, input_ids, labels=None, num_neg_samples=256,
                eval_last_only=False, **kwargs):
        if self.training and labels is not None:
            masked_input, mask_positions = self._mask_sequence(input_ids)
            seq_emb = self.bert4rec(masked_input)
        else:
            seq_emb = self.bert4rec(input_ids)
            mask_positions = None

        item_embs = self.bert4rec.item_embedding.weight[1:self.num_items + 1]

        loss = None
        logits = None

        if eval_last_only and labels is None:
            last_emb = seq_emb[:, -1, :]
            logits = torch.matmul(last_emb, item_embs.T).unsqueeze(1)
        elif labels is not None and num_neg_samples > 0 and self.num_items > num_neg_samples * 2:
            if mask_positions is not None:
                mask = (mask_positions > 0).float()
                target = (labels - 1).clamp(min=0)
            else:
                mask = (labels > 0).float()
                target = (labels - 1).clamp(min=0)

            flat_target = target.view(-1)
            flat_emb = seq_emb.view(-1, seq_emb.size(-1))
            flat_mask = mask.view(-1)

            pos_ids = flat_target[flat_mask > 0].unique()
            neg_ids = torch.randperm(self.num_items, device=input_ids.device)[:num_neg_samples]
            sampled_ids = torch.cat([pos_ids, neg_ids]).unique()

            sampled_embs = item_embs[sampled_ids]
            sampled_logits = torch.matmul(flat_emb, sampled_embs.T)

            id_to_idx = torch.zeros(self.num_items, dtype=torch.long, device=input_ids.device)
            id_to_idx[sampled_ids] = torch.arange(len(sampled_ids), device=input_ids.device)
            remapped_target = id_to_idx[flat_target]

            loss = F.cross_entropy(sampled_logits, remapped_target, reduction='none')
            loss = (loss * flat_mask).sum() / flat_mask.sum().clamp(min=1)
            logits = sampled_logits.view(seq_emb.size(0), seq_emb.size(1), -1)
        else:
            logits = torch.matmul(seq_emb, item_embs.T)
            if labels is not None:
                if mask_positions is not None:
                    mask = (mask_positions > 0).float()
                    target = (labels - 1).clamp(min=0)
                else:
                    mask = (labels > 0).float()
                    target = (labels - 1).clamp(min=0)
                loss = F.cross_entropy(
                    logits.view(-1, self.num_items),
                    target.view(-1),
                    reduction='none'
                )
                loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)

        return type('Output', (), {'loss': loss, 'logits': logits})()

    def _mask_sequence(self, input_ids):
        device = input_ids.device
        masked = input_ids.clone()
        mask_positions = torch.zeros_like(input_ids)

        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                if input_ids[i, j] == 0:
                    continue
                if torch.rand(1).item() < self.mask_prob:
                    mask_positions[i, j] = 1
                    r = torch.rand(1).item()
                    if r < 0.8:
                        masked[i, j] = self.bert4rec.mask_token
                    elif r < 0.9:
                        masked[i, j] = torch.randint(1, self.num_items + 1, (1,)).item()

        return masked, mask_positions

    @property
    def config(self):
        return type('Config', (), {'hidden_size': self.bert4rec.d_model})()
