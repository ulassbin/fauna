import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ActionTransformer(nn.Module):
    def __init__(self, input_dim=512, d_model=512, nhead=4, num_layers=2,
                 num_classes=140, dropout=0.1, max_len=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 16, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _encode(self, x, mask):
        x = self.proj(x)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=~mask)  # [B, T, d_model]

    def forward(self, x, mask):
        """
        x:    [B, T, input_dim]
        mask: [B, T]  True = valid token, False = padding
        """
        enc = self._encode(x, mask)
        m = mask.float().unsqueeze(-1)                      # [B, T, 1]
        pooled = (enc * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.norm(pooled))                 # [B, num_classes]

    def forward_temporal(self, x, mask):
        """
        Same as forward but returns per-token logits before pooling.
        Returns [B, T, num_classes] — sigmoid of this gives per-clip activations.
        """
        enc = self._encode(x, mask)
        return self.head(self.norm(enc))                    # [B, T, num_classes]