import argparse
import json
import logging
import math
import sys
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import authorship.parsing.constants as constants
from authorship.config import Config, set_seed
from authorship.parsing.book_renamer import (
    UkrainianBookRenamer,
)
from authorship.parsing.parser import (
    training_books,
    validation_books,
)
from authorship.predict import (
    AuthorVerifier,
)
from authorship.reporting import plot_command
from authorship.train import train_model

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(constants.__file__).resolve().parents[3]


def load_fragments(path: Path) -> list[dict[str, str]]:
    """Load a text fragment from a .txt file.

    The entire file content becomes a single fragment
    (the filename stem is used as the fragment id).
    """
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty file: {path}")
        return [{"id": path.stem, "text": text}]

    raise ValueError(
        f"Unsupported file format: {suffix} (expected .txt)"
    )


def split_text(
    text: str, vocabulary: Any, max_seq_len: int
) -> list[str]:
    """Split text into chunks that fit within max_seq_len tokens.

    Uses character-level splitting guided by token count estimation.
    """
    tokens = vocabulary.tokenize(text)
    token_count = len(tokens)

    if token_count <= max_seq_len:
        return [text]

    chars_per_token = len(text) / token_count
    target_chunk_chars = int(
        max_seq_len * chars_per_token * 0.9
    )

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= target_chunk_chars:
            chunks.append(remaining.strip())
            break

        # Find whitespace boundary near target size
        split_pos = target_chunk_chars
        while (
            split_pos > 0
            and not remaining[split_pos].isspace()
        ):
            split_pos -= 1

        if split_pos == 0:
            # No whitespace found — force split at target
            split_pos = target_chunk_chars

        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].strip()

    return chunks


def aggregate_scores(
    scores: list[float], method: str
) -> float:
    """Aggregate pairwise scores using the specified method."""
    if not scores:
        return 0.0

    if method == "arithmetic":
        return sum(scores) / len(scores)
    elif method == "geometric":
        # Clamp to avoid log(0)
        clamped = [max(s, 1e-10) for s in scores]
        return math.exp(
            sum(math.log(s) for s in clamped) / len(clamped)
        )
    elif method == "harmonic":
        clamped = [max(s, 1e-10) for s in scores]
        return len(clamped) / sum(1.0 / s for s in clamped)
    elif method == "median":
        return median(scores)
    else:
        raise ValueError(
            f"Unknown aggregation method: {method}"
        )


def compare_fragments(
    frag_a: dict[str, str],
    frag_b: dict[str, str],
    verifier: Any,
    vocabulary: Any,
    max_seq_len: int,
    aggregation: str,
    progress: bool = False,
) -> dict:
    """Compare two fragments, splitting into chunks if needed."""
    chunks_a = split_text(
        frag_a["text"], vocabulary, max_seq_len
    )
    chunks_b = split_text(
        frag_b["text"], vocabulary, max_seq_len
    )

    tokens_a = len(vocabulary.tokenize(frag_a["text"]))
    tokens_b = len(vocabulary.tokenize(frag_b["text"]))

    total_chunk_pairs = len(chunks_a) * len(chunks_b)
    if progress:
        width = len(str(total_chunk_pairs))
        print(
            f"  Chunks: A={len(chunks_a)}, B={len(chunks_b)}"
            f" \u2192 {total_chunk_pairs} pairwise comparison(s)",
            file=sys.stderr,
        )

    pairwise_scores = []
    pair_idx = 0
    for ca in chunks_a:
        for cb in chunks_b:
            _, prob = verifier.verify(ca, cb)
            pairwise_scores.append(round(prob, 4))
            if progress:
                pair_idx += 1
                pct = pair_idx / total_chunk_pairs * 100
                print(
                    f"\r  Pair [{pair_idx:{width}}/{total_chunk_pairs}] {pct:5.1f}%",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    if progress:
        print(file=sys.stderr)

    aggregated = aggregate_scores(
        pairwise_scores, aggregation
    )

    return {
        "fragment_a": {
            "id": frag_a["id"],
            "tokens": tokens_a,
            "chunks": len(chunks_a),
        },
        "fragment_b": {
            "id": frag_b["id"],
            "tokens": tokens_b,
            "chunks": len(chunks_b),
        },
        "pairwise_scores": pairwise_scores,
        "aggregated_score": round(aggregated, 4),
    }


def format_text_output(
    comparisons: list[dict],
    threshold: float,
    aggregation: str,
    checkpoint: str,
) -> str:
    """Format comparison results as human-readable text."""
    lines = [
        "=== Author Verification ===",
        f"Checkpoint: {checkpoint}",
        f"Threshold:  {threshold} | Aggregation: {aggregation}",
        "",
    ]

    same_count = 0
    diff_count = 0
    total = len(comparisons)

    for i, comp in enumerate(comparisons):
        is_same = comp["aggregated_score"] > threshold
        if is_same:
            same_count += 1
        else:
            diff_count += 1

        fa = comp["fragment_a"]
        fb = comp["fragment_b"]
        lines.append(f"--- Comparison {i + 1}/{total} ---")
        lines.append(
            f'  A: "{fa["id"]}" ({fa["tokens"]} tokens \u2192 {fa["chunks"]} chunks)'
        )
        lines.append(
            f'  B: "{fb["id"]}" ({fb["tokens"]} tokens \u2192 {fb["chunks"]} chunks)'
        )
        lines.append(
            f"  Pairwise: {comp['pairwise_scores']}"
        )
        lines.append(f"  Score: {comp['aggregated_score']}")
        verdict = (
            "SAME AUTHOR \u2713"
            if is_same
            else "DIFFERENT AUTHORS \u2717"
        )
        lines.append(f"  Result: {verdict}")
        lines.append("")

    lines.append(
        f"=== Summary: {same_count} same, {diff_count} different ==="
    )
    return "\n".join(lines)


def format_json_output(
    comparisons: list[dict], threshold: float
) -> str:
    """Format comparison results as JSON."""
    for comp in comparisons:
        comp["is_same_author"] = (
            comp["aggregated_score"] > threshold
        )
    return json.dumps(
        {"comparisons": comparisons},
        ensure_ascii=False,
        indent=2,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="authorship",
        description="Ukrainian authorship verification — download the corpus, train the model, compare fragments.",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download, clean and rename the book corpus",
    )
    download_parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Download the training books dataset",
    )
    download_parser.add_argument(
        "-v",
        "--validation",
        action="store_true",
        help="Download the validation books dataset",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download every URL, bypassing the download cache",
    )
    download_parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all files in the download directories before downloading",
    )
    download_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )

    rename_parser = subparsers.add_parser(
        "rename",
        help="Rename already-downloaded books to the surname-name-N format via Ollama",
    )
    rename_parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Rename books in the training directory",
    )
    rename_parser.add_argument(
        "-v",
        "--validation",
        action="store_true",
        help="Rename books in the validation directory",
    )
    rename_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-rename every book, bypassing the rename cache",
    )
    rename_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train the Siamese BiLSTM model on the downloaded corpus",
    )
    train_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: experiments/default.yaml)",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Render training metrics from the XML report to a figure",
    )
    plot_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to the training report XML (default: experiments/training_report.xml)",
    )
    plot_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Figure output path (default: docs/assets/training_curves.png)",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Compare text fragments with a trained model",
    )
    verify_parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input .txt files. Each file becomes one fragment; files are compared pairwise.",
    )
    verify_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (default: experiments/siamese_full_package.pth)",
    )
    verify_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: experiments/default.yaml)",
    )
    verify_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override same-author threshold (default: from config, 0.65)",
    )
    verify_parser.add_argument(
        "--aggregation",
        choices=[
            "arithmetic",
            "geometric",
            "harmonic",
            "median",
        ],
        default="arithmetic",
        help="Score aggregation method (default: arithmetic)",
    )
    verify_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    verify_parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write results to a file instead of stdout",
    )
    verify_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show comparison progress and chunk counts on stderr",
    )
    verify_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser


def download_command(args: argparse.Namespace) -> None:
    """Run the corpus download pipeline."""
    if not args.train and not args.validation:
        logger.error(
            "Select at least one dataset: --train or --validation"
        )
        sys.exit(1)

    if args.train:
        training_books(force=args.force, clean=args.clean)
    if args.validation:
        validation_books(force=args.force, clean=args.clean)


def rename_command(args: argparse.Namespace) -> None:
    """Rename downloaded books without downloading anything."""
    if not args.train and not args.validation:
        logger.error(
            "Select at least one dataset: --train or --validation"
        )
        sys.exit(1)

    renamer = UkrainianBookRenamer()

    if args.train:
        if not constants.SAVE_FOLDER_UKRLIB.is_dir():
            logger.error(
                "Training directory not found: %s — run 'authorship download --train' first",
                constants.SAVE_FOLDER_UKRLIB,
            )
            sys.exit(1)
        renamer.rename_books(
            constants.SAVE_FOLDER_UKRLIB, force=args.force
        )

    if args.validation:
        if not constants.SAVE_FOLDER_UKRLIB_TEST.is_dir():
            logger.error(
                "Validation directory not found: %s — run 'authorship download --validation' first",
                constants.SAVE_FOLDER_UKRLIB_TEST,
            )
            sys.exit(1)
        renamer.rename_books(
            constants.SAVE_FOLDER_UKRLIB_TEST,
            force=args.force,
        )


def train_command(args: argparse.Namespace) -> None:
    """Run model training."""
    config_path = (
        args.config
        or PROJECT_ROOT / "experiments" / "default.yaml"
    )
    config = Config.from_file(config_path)
    set_seed(config.seed)
    logger.info(
        "Device: %s | Seed: %d", config.device, config.seed
    )
    try:
        train_model(config)
    except FileNotFoundError as e:
        logger.error(
            "%s — run 'authorship download --train' first",
            e,
        )
        sys.exit(1)


def verify_command(args: argparse.Namespace) -> None:  # noqa: C901
    """Compare text fragments with a trained model."""
    # Resolve checkpoint and config paths
    checkpoint_path = (
        args.checkpoint
        or PROJECT_ROOT
        / "experiments"
        / "siamese_full_package.pth"
    )
    config_path = (
        args.config
        or PROJECT_ROOT / "experiments" / "default.yaml"
    )

    if not checkpoint_path.exists():
        logger.error(
            "Checkpoint not found: %s", checkpoint_path
        )
        sys.exit(1)

    config = Config.from_file(config_path)
    set_seed(config.seed)

    threshold = (
        args.threshold
        if args.threshold is not None
        else config.inference.threshold
    )

    # Validate input files
    for fpath in args.files:
        if not fpath.exists():
            logger.error("File not found: %s", fpath)
            sys.exit(1)
        if fpath.suffix.lower() != ".txt":
            logger.error(
                "Unsupported file format: %s (expected .txt)",
                fpath,
            )
            sys.exit(1)

    # Load all fragments
    all_fragments: list[dict[str, str]] = []
    for fpath in args.files:
        try:
            frags = load_fragments(fpath)
            all_fragments.extend(frags)
            logger.info(
                "Loaded %d fragment(s) from %s",
                len(frags),
                fpath,
            )
        except ValueError as e:
            logger.error("Error loading %s: %s", fpath, e)
            sys.exit(1)

    if len(all_fragments) < 2:
        logger.error(
            "At least 2 fragments are required for comparison (got %d)",
            len(all_fragments),
        )
        sys.exit(1)

    # Load model
    logger.info(
        "Loading model from %s ...", checkpoint_path
    )

    verifier = AuthorVerifier(checkpoint_path, config)
    vocabulary = verifier.vocabulary
    max_seq_len = config.data.max_seq_len

    # Run pairwise comparisons
    pairs = list(combinations(range(len(all_fragments)), 2))
    logger.info("Running %d comparison(s)...", len(pairs))

    comparisons = []
    total_pairs = len(pairs)
    for i, (idx_a, idx_b) in enumerate(pairs):
        if args.progress:
            fa = all_fragments[idx_a]
            fb = all_fragments[idx_b]
            print(
                f'[{i + 1}/{total_pairs}] Comparing "{fa["id"]}" vs "{fb["id"]}"...',
                file=sys.stderr,
            )
        result = compare_fragments(
            all_fragments[idx_a],
            all_fragments[idx_b],
            verifier,
            vocabulary,
            max_seq_len,
            args.aggregation,
            progress=args.progress,
        )
        comparisons.append(result)

    # Format results
    if args.output == "json":
        output_str = format_json_output(
            comparisons, threshold
        )
    else:
        output_str = format_text_output(
            comparisons,
            threshold,
            args.aggregation,
            str(checkpoint_path),
        )

    # Write results
    if args.output_file:
        args.output_file.write_text(
            output_str, encoding="utf-8"
        )
        logger.info(
            "Results written to %s", args.output_file
        )
    else:
        print(output_str)


_COMMANDS = {
    "download": download_command,
    "rename": rename_command,
    "train": train_command,
    "plot": plot_command,
    "verify": verify_command,
}


def main() -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    verbose = getattr(args, "verbose", False) or getattr(
        args, "debug", False
    )
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
