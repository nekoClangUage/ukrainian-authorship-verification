import torch
import torch.nn as nn


class SiameseLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        classifier_h1: int = 512,
        classifier_h2: int = 256,
        dropout: float = 0.3,
        embed_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.embedding: nn.Embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.embed_dropout: nn.Dropout = nn.Dropout(
            p=embed_dropout
        )

        self.lstm: nn.LSTM = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Output size will be (hidden * 2 <- bidirectional)
        self.encoder_out_dim: int = hidden_dim * 2

        # Classifier head
        # Input: 4 * encoder_out (u, v, |u-v|, u*v)
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(
                self.encoder_out_dim * 4, classifier_h1
            ),
            nn.LayerNorm(classifier_h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_h1, classifier_h2),
            nn.LayerNorm(classifier_h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_h2, 1),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        embedded: torch.Tensor = self.embed_dropout(
            self.embedding(x)
        )
        lstm_out, _ = self.lstm(embedded)

        # Global average pooling (padding positions ignored)
        mask: torch.Tensor = (x != 0).unsqueeze(-1).float()
        pooled: torch.Tensor = (lstm_out * mask).sum(
            dim=1
        ) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled

    def forward(
        self, text1: torch.Tensor, text2: torch.Tensor
    ) -> torch.Tensor:
        u: torch.Tensor = self.forward_one(text1)
        v: torch.Tensor = self.forward_one(text2)

        diff: torch.Tensor = torch.abs(u - v)
        prod: torch.Tensor = u * v
        combined: torch.Tensor = torch.cat(
            [u, v, diff, prod], dim=1
        )

        return self.classifier(combined)
