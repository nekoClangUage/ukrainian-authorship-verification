import logging
from pathlib import Path

import torch

from authorship.config import Config
from authorship.model import SiameseLSTM
from authorship.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class AuthorVerifier:
    def __init__(
        self,
        checkpoint_path: Path | str,
        config: Config | None = None,
    ) -> None:
        self.configuration = config or Config()

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.configuration.device,
        )

        self.word2idx = checkpoint["vocab_word2idx"]
        self.idx2word = {
            value: key
            for key, value in self.word2idx.items()
        }
        self.vocabulary = Vocabulary(
            self.configuration,
            self.word2idx,
            self.idx2word,
            len(self.word2idx),
        )

        self.model = SiameseLSTM(
            vocab_size=len(self.word2idx),
            embed_dim=checkpoint["config"]["embed_dim"],
            hidden_dim=checkpoint["config"]["hidden_dim"],
            classifier_h1=checkpoint["config"].get(
                "classifier_h1", 512
            ),
            classifier_h2=checkpoint["config"].get(
                "classifier_h2", 256
            ),
            dropout=checkpoint["config"].get(
                "dropout", 0.3
            ),
        ).to(self.configuration.device)
        self.model.load_state_dict(
            checkpoint["model_state"]
        )
        self.model.eval()

        logger.info(
            "Model loaded from %s (vocab_size=%d, embed=%d, hidden=%d)",
            checkpoint_path,
            len(self.word2idx),
            checkpoint["config"]["embed_dim"],
            checkpoint["config"]["hidden_dim"],
        )

    def _preprocess(self, text: str) -> torch.Tensor:
        indices: torch.Tensor = (
            self.vocabulary.text_to_indices(text)
        )
        return indices.unsqueeze(0)  # (1, seq_len)

    def verify(
        self, text1: str, text2: str
    ) -> tuple[bool, float]:
        t1 = self._preprocess(text1).to(
            self.configuration.device
        )
        t2 = self._preprocess(text2).to(
            self.configuration.device
        )

        with torch.no_grad():
            logits = self.model(t1, t2)
            probability = torch.sigmoid(logits).item()

        is_same = (
            probability
            > self.configuration.inference.threshold
        )
        return is_same, probability
