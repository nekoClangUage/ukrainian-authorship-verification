import json
from pathlib import Path
from statistics import median

import pytest

from authorship.cli import (
    aggregate_scores,
    build_parser,
    format_json_output,
    format_text_output,
    load_fragments,
    rename_command,
    split_text,
)
from authorship.reporting import (
    _read_epochs,
    plot_training_report,
)


# ---------------------------------------------------------------------------
# load_fragments
# ---------------------------------------------------------------------------
class TestLoadFragments:
    def test_txt_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Привіт, це текст.", encoding="utf-8")

        result = load_fragments(f)

        assert result == [
            {"id": "sample", "text": "Привіт, це текст."}
        ]

    def test_empty_txt_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Empty file"):
            load_fragments(f)

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported"):
            load_fragments(f)

    def test_json_rejected(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"fragments": []}', encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported"):
            load_fragments(f)


# ---------------------------------------------------------------------------
# split_text — uses a lightweight mock vocabulary
# ---------------------------------------------------------------------------
class MockVocabulary:
    """Minimal mock: tokenize splits on whitespace."""

    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


class TestSplitText:
    def test_short_text_single_chunk(self):
        vocab = MockVocabulary()
        text = "Коротке речення з кількох слів"

        result = split_text(text, vocab, max_seq_len=100)

        assert result == [text]

    def test_long_text_multiple_chunks(self):
        vocab = MockVocabulary()
        words = ["слово"] * 200
        text = " ".join(words)

        result = split_text(text, vocab, max_seq_len=10)

        assert len(result) > 1

    def test_no_empty_chunks(self):
        vocab = MockVocabulary()
        words = ["текст"] * 100
        text = " ".join(words)

        result = split_text(text, vocab, max_seq_len=5)

        for chunk in result:
            assert len(chunk.strip()) > 0


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------
class TestAggregateScores:
    def test_arithmetic(self):
        assert aggregate_scores(
            [0.4, 0.6], "arithmetic"
        ) == pytest.approx(0.5)

    def test_geometric(self):
        assert aggregate_scores(
            [0.25, 1.0], "geometric"
        ) == pytest.approx(0.5)

    def test_harmonic(self):
        result = aggregate_scores([0.4, 0.6], "harmonic")
        expected = 2 / (1 / 0.4 + 1 / 0.6)
        assert result == pytest.approx(expected)

    def test_median_odd(self):
        assert aggregate_scores(
            [0.1, 0.5, 0.9], "median"
        ) == pytest.approx(0.5)

    def test_median_even(self):
        result = aggregate_scores(
            [0.2, 0.4, 0.6, 0.8], "median"
        )
        assert result == pytest.approx(
            median([0.2, 0.4, 0.6, 0.8])
        )

    def test_empty_list(self):
        assert aggregate_scores([], "arithmetic") == 0.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            aggregate_scores([0.5], "unknown_method")


# ---------------------------------------------------------------------------
# format_text_output
# ---------------------------------------------------------------------------
class TestFormatTextOutput:
    @staticmethod
    def _make_comparison(score):
        return {
            "fragment_a": {
                "id": "a",
                "tokens": 50,
                "chunks": 1,
            },
            "fragment_b": {
                "id": "b",
                "tokens": 60,
                "chunks": 1,
            },
            "pairwise_scores": [score],
            "aggregated_score": score,
        }

    def test_same_author(self):
        comp = self._make_comparison(0.8)
        output = format_text_output(
            [comp],
            threshold=0.5,
            aggregation="arithmetic",
            checkpoint="test.pth",
        )

        assert "SAME AUTHOR" in output

    def test_different_authors(self):
        comp = self._make_comparison(0.3)
        output = format_text_output(
            [comp],
            threshold=0.5,
            aggregation="arithmetic",
            checkpoint="test.pth",
        )

        assert "DIFFERENT AUTHORS" in output

    def test_summary_line(self):
        comp_same = self._make_comparison(0.9)
        comp_diff = self._make_comparison(0.1)
        output = format_text_output(
            [comp_same, comp_diff],
            threshold=0.5,
            aggregation="arithmetic",
            checkpoint="test.pth",
        )

        assert "1 same, 1 different" in output


# ---------------------------------------------------------------------------
# format_json_output
# ---------------------------------------------------------------------------
class TestFormatJsonOutput:
    @staticmethod
    def _make_comparison(score):
        return {
            "fragment_a": {
                "id": "a",
                "tokens": 50,
                "chunks": 1,
            },
            "fragment_b": {
                "id": "b",
                "tokens": 60,
                "chunks": 1,
            },
            "pairwise_scores": [score],
            "aggregated_score": score,
        }

    def test_valid_json(self):
        comp = self._make_comparison(0.7)
        output = format_json_output([comp], threshold=0.5)

        parsed = json.loads(output)
        assert "comparisons" in parsed

    def test_is_same_author_true(self):
        comp = self._make_comparison(0.8)
        output = format_json_output([comp], threshold=0.5)

        parsed = json.loads(output)
        assert (
            parsed["comparisons"][0]["is_same_author"]
            is True
        )

    def test_is_same_author_false(self):
        comp = self._make_comparison(0.3)
        output = format_json_output([comp], threshold=0.5)

        parsed = json.loads(output)
        assert (
            parsed["comparisons"][0]["is_same_author"]
            is False
        )


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------
class TestBuildParser:
    def test_verify_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["verify", "file.txt"])

        assert args.command == "verify"
        assert args.aggregation == "arithmetic"
        assert args.output == "text"
        assert args.verbose is False
        assert args.checkpoint is None
        assert args.threshold is None

    def test_verify_all_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "verify",
                "--checkpoint",
                "model.pth",
                "--config",
                "cfg.yaml",
                "--threshold",
                "0.7",
                "--aggregation",
                "geometric",
                "--output",
                "json",
                "-v",
                "a.txt",
                "b.txt",
            ]
        )

        assert str(args.checkpoint) == "model.pth"
        assert str(args.config) == "cfg.yaml"
        assert args.threshold == 0.7
        assert args.aggregation == "geometric"
        assert args.output == "json"
        assert args.verbose is True
        assert len(args.files) == 2

    def test_download_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "download",
                "--train",
                "--validation",
                "--force",
                "--clean",
            ]
        )

        assert args.command == "download"
        assert args.train is True
        assert args.validation is True
        assert args.force is True
        assert args.clean is True

    def test_rename_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["rename", "--train"])

        assert args.command == "rename"
        assert args.train is True
        assert args.validation is False
        assert args.force is False
        assert args.debug is False

    def test_rename_all_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "rename",
                "--train",
                "--validation",
                "--force",
                "--debug",
            ]
        )

        assert args.command == "rename"
        assert args.train is True
        assert args.validation is True
        assert args.force is True
        assert args.debug is True

    def test_rename_requires_dataset(self):
        parser = build_parser()
        args = parser.parse_args(["rename"])

        with pytest.raises(SystemExit):
            rename_command(args)

    def test_rename_missing_directory_exits(
        self, monkeypatch
    ):
        parser = build_parser()
        args = parser.parse_args(["rename", "--train"])

        monkeypatch.setattr(
            "authorship.parsing.constants.SAVE_FOLDER_UKRLIB",
            Path("/nonexistent/ukr_data"),
        )

        with pytest.raises(SystemExit):
            rename_command(args)

    def test_train_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["train"])

        assert args.command == "train"
        assert args.config is None

    def test_plot_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["plot"])

        assert args.command == "plot"
        assert args.report is None
        assert args.output is None


# ---------------------------------------------------------------------------
# reporting — training report → figure
# ---------------------------------------------------------------------------
_REPORT_XML = """<?xml version='1.0' encoding='utf-8'?>
<training_report status="completed">
  <epochs>
    <epoch number="1">
      <train_loss>0.6</train_loss>
      <val_loss>0.5</val_loss>
      <val_acc>78.0</val_acc>
      <val_f1>0.75</val_f1>
      <val_precision>0.86</val_precision>
      <val_recall>0.67</val_recall>
      <val_auc_roc>0.88</val_auc_roc>
      <lr>0.0005</lr>
      <avg_grad_norm>1.5</avg_grad_norm>
    </epoch>
    <epoch number="2">
      <train_loss>0.2</train_loss>
      <val_loss>0.3</val_loss>
      <val_acc>87.0</val_acc>
      <val_f1>0.87</val_f1>
      <val_precision>0.88</val_precision>
      <val_recall>0.86</val_recall>
      <val_auc_roc>0.95</val_auc_roc>
      <lr>0.0005</lr>
      <avg_grad_norm>1.2</avg_grad_norm>
    </epoch>
  </epochs>
</training_report>
"""


class TestReporting:
    def _write_report(self, tmp_path):
        report = tmp_path / "report.xml"
        report.write_text(_REPORT_XML, encoding="utf-8")
        return report

    def test_read_epochs(self, tmp_path):
        report = self._write_report(tmp_path)

        epochs = _read_epochs(report)

        assert len(epochs) == 2
        assert epochs[0]["train_loss"] == pytest.approx(0.6)
        assert epochs[1]["val_auc_roc"] == pytest.approx(
            0.95
        )

    def test_read_epochs_empty_raises(self, tmp_path):
        report = tmp_path / "empty.xml"
        report.write_text(
            "<training_report/>", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="No <epoch>"):
            _read_epochs(report)

    def test_plot_creates_png(self, tmp_path):
        report = self._write_report(tmp_path)
        output = tmp_path / "fig.png"

        result = plot_training_report(report, output)

        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0
        assert (
            output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        )

    def test_no_command_exits(self):
        parser = build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])
