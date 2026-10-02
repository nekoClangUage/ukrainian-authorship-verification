import itertools
import random
import sys

import pytest

from authorship import baseline, train
from authorship.baseline import (
    evaluate_pairs,
    tfidf_cosine_scores,
)
from authorship.config import Config, set_seed
from authorship.pairs import (
    SiamesePairGenerator,
    generate_split_pairs,
    split_author_chunks,
)

AUTHOR_A = [
    f"alpha beta gamma delta {i} " * 10 for i in range(6)
]
AUTHOR_B = [
    f"xray yankee zulu omega {i} " * 10 for i in range(6)
]


def _same_and_cross_pairs():
    same = list(itertools.pairwise(AUTHOR_A)) + list(
        itertools.pairwise(AUTHOR_B)
    )
    cross = list(
        zip(AUTHOR_A[:-1], AUTHOR_B[1:], strict=True)
    )
    return same, cross


class TestTfidfCosineScores:
    def test_same_author_scores_above_cross_author(self):
        same, cross = _same_and_cross_pairs()
        scores, _ = tfidf_cosine_scores(same + cross)

        same_scores = scores[: len(same)]
        cross_scores = scores[len(same) :]
        assert same_scores.min() > cross_scores.max()

    def test_one_score_per_pair(self):
        same, cross = _same_and_cross_pairs()
        scores, _ = tfidf_cosine_scores(same + cross)

        assert len(scores) == len(same) + len(cross)

    def test_scores_in_unit_range(self):
        same, cross = _same_and_cross_pairs()
        scores, _ = tfidf_cosine_scores(same + cross)

        assert ((scores >= 0.0) & (scores <= 1.0)).all()


class TestEvaluatePairs:
    def test_perfect_separation_metrics(self):
        same, cross = _same_and_cross_pairs()
        labels = [1.0] * len(same) + [0.0] * len(cross)

        metrics = evaluate_pairs(
            same + cross, labels, threshold=0.5
        )

        assert metrics["accuracy"] == pytest.approx(100.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["auc_roc"] == pytest.approx(1.0)
        assert (
            metrics["mean_same_author_score"]
            > metrics["mean_different_author_score"]
        )


def test_split_pairs_preserve_original_training_sequence():
    authors = {
        name: [f"{name} chunk {i}" for i in range(20)]
        for name in ("alpha", "beta", "gamma")
    }
    random.seed(2026)
    expected_train = {}
    expected_val = {}
    for author, chunks in authors.items():
        texts = list(chunks)
        random.shuffle(texts)
        split_idx = int(len(texts) * 0.8)
        expected_train[author] = texts[:split_idx]
        expected_val[author] = texts[split_idx:]
    train_pairs, train_labels = (
        SiamesePairGenerator.generate(expected_train, 7)
    )
    val_pairs, val_labels = SiamesePairGenerator.generate(
        expected_val, 5
    )

    random.seed(2026)
    actual_train, actual_val = split_author_chunks(authors)
    assert actual_train == expected_train
    assert actual_val == expected_val
    assert generate_split_pairs(
        actual_train, actual_val, 7, 5
    ) == (
        train_pairs,
        train_labels,
        val_pairs,
        val_labels,
    )
    for author in authors:
        assert set(actual_train[author]).isdisjoint(
            actual_val[author]
        )
        assert authors[author][0] == f"{author} chunk 0"


@pytest.mark.parametrize("override", [False, True])
def test_baseline_matches_training_validation_and_fits_only_train(
    tmp_path, monkeypatch, override
):
    """Compare the actual CLI path with training's DataLoader input."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for author in ("alpha", "beta", "gamma"):
        chunks = [
            f"{author} marker{author}{chr(97 + i)}".ljust(
                64
            )
            + "\n"
            for i in range(20)
        ]
        (corpus / f"{author}-1.txt").write_text(
            "".join(chunks), encoding="utf-8"
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 31\ndata:\n"
        "  max_seq_len: 1\n"
        "  min_freq: 1\n"
        "  divider_target_count: 20\n"
        "  train_samples_per_author: 7\n"
        "  val_samples_per_author: 5\n"
        "  vocab_force_rebuild: true\n",
        encoding="utf-8",
    )
    config = Config.from_file(config_path)
    argv = [
        "baseline",
        str(corpus),
        "--config",
        str(config_path),
    ]
    if override:
        argv += [
            "--seed",
            "42",
            "--samples",
            "3",
            "--train-samples",
            "4",
            "--target-count",
            "10",
        ]
        config.seed = 42
        config.data.val_samples_per_author = 3
        config.data.train_samples_per_author = 4
        config.data.divider_target_count = 10

    monkeypatch.setattr(
        train.constants, "SAVE_FOLDER_UKRLIB", corpus
    )
    monkeypatch.setattr(
        train, "VOCAB_CACHE", tmp_path / "vocab.json"
    )
    datasets = []

    class DataPrepared(Exception):
        pass

    def capture_loader(dataset, **kwargs):
        datasets.append(dataset)
        if len(datasets) == 2:
            raise DataPrepared
        return None

    monkeypatch.setattr(train, "DataLoader", capture_loader)
    set_seed(config.seed)
    with pytest.raises(DataPrepared):
        train.train_model(config)

    real_evaluate = baseline.evaluate_pairs
    evaluated = []

    def capture_evaluation(
        pairs, labels, threshold, vectorizer
    ):
        assert pairs == datasets[1].pairs
        assert labels == datasets[1].labels
        assert (
            len(pairs)
            == 2 * 3 * config.data.val_samples_per_author
        )
        analyzer = vectorizer.build_analyzer()
        train_tokens = {
            token
            for pair in datasets[0].pairs
            for text in pair
            for token in analyzer(text)
        }
        val_tokens = {
            token
            for pair in pairs
            for text in pair
            for token in analyzer(text)
        }
        held_out_tokens = val_tokens - train_tokens
        assert held_out_tokens
        assert held_out_tokens.isdisjoint(
            vectorizer.vocabulary_
        )
        evaluated.append(
            real_evaluate(
                pairs, labels, threshold, vectorizer
            )
        )
        return evaluated[-1]

    monkeypatch.setattr(
        baseline, "evaluate_pairs", capture_evaluation
    )
    monkeypatch.setattr(sys, "argv", argv)
    baseline.main()
    assert len(evaluated) == 1
