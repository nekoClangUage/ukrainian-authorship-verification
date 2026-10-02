import logging
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from authorship.exceptions import InsufficientAuthorsError
from authorship.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def split_author_chunks(
    authors_data: dict[Any, list[str]],
) -> tuple[dict[Any, list[str]], dict[Any, list[str]]]:
    """Apply the training pipeline's shuffled 80/20 split per author.

    Seed Python's random generator before calling. Input lists are copied
    so that repeated preparation does not shuffle already-shuffled data.
    """
    train_data = {}
    val_data = {}
    for author, chunks in authors_data.items():
        texts = list(chunks)
        random.shuffle(texts)
        split_idx = int(len(texts) * 0.8)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        if not val_texts:
            logger.warning(
                "Author '%s' has no validation texts — duplicating train set.",
                author,
            )
            val_texts = train_texts
        train_data[author] = train_texts
        val_data[author] = val_texts
    return train_data, val_data


def generate_split_pairs(
    train_data: dict[Any, list[str]],
    val_data: dict[Any, list[str]],
    train_samples_per_author: int,
    val_samples_per_author: int,
) -> tuple[
    list[tuple[str, str]],
    list[float],
    list[tuple[str, str]],
    list[float],
]:
    """Generate train then validation pairs, preserving training's RNG order.

    Validation pairs depend on the preceding train-pair generation, even
    when only validation metrics are needed by the caller.
    """
    train_pairs, train_labels = (
        SiamesePairGenerator.generate(
            train_data, train_samples_per_author
        )
    )
    val_pairs, val_labels = SiamesePairGenerator.generate(
        val_data, val_samples_per_author
    )
    return train_pairs, train_labels, val_pairs, val_labels


# ==========================================
# PAIR GENERATOR
# ==========================================
class SiamesePairGenerator:
    @staticmethod
    def generate(
        authors_data: dict[Any, list[str]],
        samples_per_author: int,
    ) -> tuple[list[tuple[str, str]], list[float]]:

        pairs: list[tuple[str, str]] = []
        labels: list[float] = []

        author_ids: list[Any] = list(authors_data.keys())

        if len(author_ids) < 2:
            raise InsufficientAuthorsError(
                f"At least 2 authors are required to generate pairs, got: {len(author_ids)}"
            )

        logger.info(
            "Generating pairs for %d authors...",
            len(author_ids),
        )

        # 1. Positive pairs (Same author) -> Label 1
        for auth in author_ids:
            texts: list[str] = authors_data[auth]
            # Take samples as much as we can
            count: int = samples_per_author

            for _ in range(count):
                # Two different texts of one author
                t1, t2 = random.choices(texts, k=2)
                pairs.append((t1, t2))
                labels.append(1.0)

        # Count of positive examples (for future negative pairs count important)
        positive_count: int = len(pairs)

        # 2. Negative pairs (Different authors) -> Label 0
        # (Positive and negative should be 50/50)
        current_neg: int = 0
        while current_neg < positive_count:
            # Take 2 different authors
            a1, a2 = random.sample(author_ids, 2)

            t1: str = random.choice(authors_data[a1])
            t2: str = random.choice(authors_data[a2])

            pairs.append((t1, t2))
            labels.append(0.0)
            current_neg += 1

        combined = list(zip(pairs, labels, strict=True))
        random.shuffle(combined)

        # zip(*combined) returns tuples
        pairs_shuffled, labels_shuffled = zip(
            *combined, strict=True
        )

        # so type conversion to the list
        return list(pairs_shuffled), list(labels_shuffled)


class AuthorPairDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        labels: list[float],
        vocab: Vocabulary,
    ) -> None:
        self.pairs: list[tuple[str, str]] = pairs
        self.labels: list[float] = labels
        self.vocab: Vocabulary = vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text1, text2 = self.pairs[idx]
        label = self.labels[idx]

        seq1: torch.Tensor = self.vocab.text_to_indices(
            text1
        )
        seq2: torch.Tensor = self.vocab.text_to_indices(
            text2
        )

        return (
            seq1,
            seq2,
            torch.tensor(label, dtype=torch.float32),
        )
