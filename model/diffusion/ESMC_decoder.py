import torch.nn as nn
from .decode_Config import Config

class ESMC_decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared_encoder = nn.Sequential(
            nn.Linear(Config.feature_dim, Config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(Config.hidden_dim),
            nn.Dropout(Config.dropout_rate)
        )

        self.classifier = nn.Conv1d(
            in_channels=Config.hidden_dim,
            out_channels=Config.num_classes,
            kernel_size=1
        )

    def forward(self, x, mask=None):
        x = self.shared_encoder(x)  # (B, seq_len, hidden_dim)
        logits = self.classifier(x.transpose(1, 2))  # (B, num_classes, seq_len)
        logits = logits.transpose(1, 2)  # (B, seq_len, num_classes)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return logits