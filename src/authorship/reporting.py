"""Rendering of training artifacts into presentation figures.

This module is deliberately independent of ``train.py``: training only
produces the XML report with the metrics, and rendering that report
into figures is a separate responsibility. Nothing here needs the
model, the GPU, or the training data — only the XML file.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "experiments" / "training_report.xml"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "docs" / "assets" / "training_curves.png"
)

_ACCENT = "#2563eb"
_ORANGE = "#f59e0b"
_GREEN = "#059669"


def _read_epochs(
    report_path: Path,
) -> list[dict[str, float]]:
    """Extract per-epoch metrics from a training report XML."""
    root = ET.parse(report_path).getroot()

    epochs = []
    for ep in root.findall("./epochs/epoch"):
        epochs.append(
            {
                tag: float(ep.findtext(tag, "0"))
                for tag in (
                    "train_loss",
                    "val_loss",
                    "val_acc",
                    "val_f1",
                    "val_precision",
                    "val_recall",
                    "val_auc_roc",
                )
            }
        )
    if not epochs:
        raise ValueError(
            f"No <epoch> entries found in {report_path}"
        )
    return epochs


def plot_training_report(
    report_path: Path = DEFAULT_REPORT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Render the per-epoch metrics of a training report to a figure.

    Left panel: train/val loss. Right panel: validation metrics
    (accuracy, F1, ROC-AUC). Returns the written figure path.
    """
    report_path = Path(report_path)
    output_path = Path(output_path)

    epochs = _read_epochs(report_path)
    numbers = list(range(1, len(epochs) + 1))

    with plt.rc_context(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
        }
    ):
        fig, (ax_loss, ax_val) = plt.subplots(
            1, 2, figsize=(11, 4)
        )

        ax_loss.plot(
            numbers,
            [e["train_loss"] for e in epochs],
            marker="o",
            color=_ACCENT,
            label="train loss",
        )
        ax_loss.plot(
            numbers,
            [e["val_loss"] for e in epochs],
            marker="o",
            color=_ORANGE,
            label="val loss",
        )
        ax_loss.set_title("Cross-entropy loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xticks(numbers)
        ax_loss.legend(frameon=False)

        ax_val.plot(
            numbers,
            [e["val_acc"] / 100 for e in epochs],
            marker="o",
            color=_ACCENT,
            label="accuracy",
        )
        ax_val.plot(
            numbers,
            [e["val_f1"] for e in epochs],
            marker="s",
            color=_GREEN,
            label="F1",
        )
        ax_val.plot(
            numbers,
            [e["val_auc_roc"] for e in epochs],
            marker="^",
            color=_ORANGE,
            label="ROC-AUC",
        )
        last = epochs[-1]
        ax_val.set_title(
            "Validation metrics — "
            f"acc {last['val_acc']:.1f}%, "
            f"F1 {last['val_f1']:.3f}, "
            f"ROC-AUC {last['val_auc_roc']:.3f}"
        )
        ax_val.set_xlabel("Epoch")
        ax_val.set_ylabel("Score")
        ax_val.set_xticks(numbers)
        ax_val.legend(frameon=False, loc="lower right")

        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", output_path)
    return output_path


def plot_command(args) -> None:
    """CLI entry point for ``authorship plot``."""
    try:
        plot_training_report(
            report_path=args.report or DEFAULT_REPORT_PATH,
            output_path=args.output or DEFAULT_OUTPUT_PATH,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s — run 'authorship train' first", e)
        raise SystemExit(1) from e
