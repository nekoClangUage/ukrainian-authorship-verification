"""TF-IDF + cosine similarity baseline for authorship verification.

A deliberately simple reference point for the Siamese BiLSTM: the same
validation pairs are scored by cosine similarity of TF-IDF vectors fitted
on training chunks. The split and pair generation match the BiLSTM for
the same corpus, configuration and seed. Run as a module against a corpus
directory in the Divider filename format (surname-name-N.txt).
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity

import authorship.parsing.constants as constants
from authorship.config import Config
from authorship.pairs import (
    generate_split_pairs,
    split_author_chunks,
)
from authorship.parsing.divider import Divider

logger = logging.getLogger(__name__)


def tfidf_cosine_scores(
    pairs: list[tuple[str, str]],
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """Score every text pair with TF-IDF cosine similarity.

    Args:
        pairs: text pairs to compare.
        vectorizer: pre-fitted vectorizer; a new one is fitted on all
            pair texts when omitted.

    Returns:
        (scores, vectorizer) with one similarity value per pair.
    """
    texts = sorted(
        {text for pair in pairs for text in pair}
    )

    if vectorizer is None:
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
    else:
        matrix = vectorizer.transform(texts)

    index = {text: i for i, text in enumerate(texts)}
    scores = np.array(
        [
            cosine_similarity(
                matrix[index[a]], matrix[index[b]]
            )[0, 0]
            for a, b in pairs
        ]
    )
    return scores, vectorizer


def evaluate_pairs(
    pairs: list[tuple[str, str]],
    labels: list[float],
    threshold: float = 0.5,
    vectorizer: TfidfVectorizer | None = None,
) -> dict[str, float]:
    """Compute classification metrics for TF-IDF cosine scores."""
    scores, _ = tfidf_cosine_scores(pairs, vectorizer)
    preds = (scores > threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "mean_same_author_score": float(
            scores[np.array(labels) == 1.0].mean()
        ),
        "mean_different_author_score": float(
            scores[np.array(labels) == 0.0].mean()
        ),
        "accuracy": float(
            accuracy_score(labels, preds) * 100
        ),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
    }
    try:
        metrics["auc_roc"] = float(
            roc_auc_score(labels, scores)
        )
    except ValueError:
        metrics["auc_roc"] = 0.0

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TF-IDF + cosine on the BiLSTM's validation split and pairs.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=constants.SAVE_FOLDER_UKRLIB,
        help="Corpus directory with surname-name-N.txt files (default: training data folder)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(constants.__file__)
        .resolve()
        .parents[3]
        / "experiments"
        / "default.yaml",
        help="BiLSTM YAML config (default: experiments/default.yaml)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override validation samples per author (default: from config)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Override training samples per author; affects validation RNG state (default: from config)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Override chunks per author (default: from config)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for the same-author decision (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override split and pair-generation seed (default: from config)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config.from_file(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.samples is not None:
        config.data.val_samples_per_author = args.samples
    if args.train_samples is not None:
        config.data.train_samples_per_author = (
            args.train_samples
        )
    if args.target_count is not None:
        config.data.divider_target_count = args.target_count

    random.seed(config.seed)
    divider = Divider(
        target_count=config.data.divider_target_count,
        min_length=config.data.max_seq_len,
    )
    authors_data = divider.parse_directory(args.directory)
    train_data, val_data = split_author_chunks(authors_data)
    _, _, pairs, labels = generate_split_pairs(
        train_data,
        val_data,
        config.data.train_samples_per_author,
        config.data.val_samples_per_author,
    )
    logger.info(
        "Baseline on %d authors, %d validation pairs | Seed: %d",
        len(authors_data),
        len(pairs),
        config.seed,
    )

    vectorizer = TfidfVectorizer(sublinear_tf=True)
    vectorizer.fit(
        [
            text
            for texts in train_data.values()
            for text in texts
        ]
    )
    metrics = evaluate_pairs(
        pairs, labels, args.threshold, vectorizer=vectorizer
    )
    for name, value in metrics.items():
        logger.info("%s: %.4f", name, value)


if __name__ == "__main__":
    main()
