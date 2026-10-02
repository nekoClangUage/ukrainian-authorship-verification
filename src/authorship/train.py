import logging
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

import authorship.parsing.constants as constants
from authorship.config import (
    Config,
)
from authorship.model import SiameseLSTM
from authorship.pairs import (
    AuthorPairDataset,
    generate_split_pairs,
    split_author_chunks,
)
from authorship.parsing.divider import Divider
from authorship.vocabulary import (
    Vocabulary,
    compute_data_fingerprint,
)

logger = logging.getLogger(__name__)

# ==========================================
# PROJECT PATHS
# ==========================================
PROJECT_ROOT = Path(constants.__file__).resolve().parents[3]
VOCAB_CACHE = (
    PROJECT_ROOT / "experiments" / "vocab_cache.json"
)
CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "siamese_full_package.pth"
)
TRAINING_REPORT = (
    PROJECT_ROOT / "experiments" / "training_report.xml"
)


def _log_model_summary(model: SiameseLSTM) -> None:
    """Print a parameter summary: total, trainable, and per-layer counts."""
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    logger.info("=" * 60)
    logger.info("MODEL PARAMETER SUMMARY")
    logger.info("  Total parameters:     %d", total_params)
    logger.info(
        "  Trainable parameters: %d", trainable_params
    )
    logger.info(
        "  Non-trainable:        %d",
        total_params - trainable_params,
    )
    logger.info("-" * 60)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                "  %-45s %s  (%d)",
                name,
                str(list(param.shape)),
                param.numel(),
            )
    logger.info("=" * 60)


def _get_current_lr(optimizer: optim.Optimizer) -> float:
    """Return the current learning rate from the first param group."""
    return optimizer.param_groups[0]["lr"]


def _build_xml_report(
    history: dict[str, list[float]],
    total_params: int,
    trainable_params: int,
    config: Config,
    training_duration_seconds: float,
    status: str = "in_progress",
) -> ET.Element:
    """Build the full training report as an ElementTree Element."""
    root = ET.Element("training_report", status=status)

    # --- meta ---
    meta = ET.SubElement(root, "meta")
    ET.SubElement(
        meta, "training_duration_seconds"
    ).text = f"{training_duration_seconds:.1f}"
    ET.SubElement(meta, "seed").text = str(config.seed)
    ET.SubElement(meta, "device").text = str(config.device)

    # --- model info ---
    model_info = ET.SubElement(root, "model")
    ET.SubElement(model_info, "total_params").text = str(
        total_params
    )
    ET.SubElement(
        model_info, "trainable_params"
    ).text = str(trainable_params)

    # --- config snapshot ---
    cfg_el = ET.SubElement(root, "config")

    model_cfg = ET.SubElement(cfg_el, "model")
    ET.SubElement(model_cfg, "embed_dim").text = str(
        config.model.embed_dim
    )
    ET.SubElement(model_cfg, "hidden_dim").text = str(
        config.model.hidden_dim
    )
    ET.SubElement(model_cfg, "classifier_h1").text = str(
        config.model.classifier_h1
    )
    ET.SubElement(model_cfg, "classifier_h2").text = str(
        config.model.classifier_h2
    )
    ET.SubElement(model_cfg, "dropout").text = str(
        config.model.dropout
    )
    ET.SubElement(model_cfg, "embed_dropout").text = str(
        config.model.embed_dropout
    )

    training_cfg = ET.SubElement(cfg_el, "training")
    ET.SubElement(training_cfg, "lr").text = str(
        config.training.lr
    )
    ET.SubElement(training_cfg, "epochs").text = str(
        config.training.epochs
    )
    ET.SubElement(training_cfg, "batch_size").text = str(
        config.training.batch_size
    )
    ET.SubElement(
        training_cfg, "scheduler_factor"
    ).text = str(config.training.scheduler_factor)
    ET.SubElement(
        training_cfg, "scheduler_patience"
    ).text = str(config.training.scheduler_patience)
    ET.SubElement(
        training_cfg, "scheduler_min_lr"
    ).text = str(config.training.scheduler_min_lr)
    ET.SubElement(
        training_cfg, "grad_clip_max_norm"
    ).text = str(config.training.grad_clip_max_norm)
    ET.SubElement(training_cfg, "weight_decay").text = str(
        config.training.weight_decay
    )

    data_cfg = ET.SubElement(cfg_el, "data")
    ET.SubElement(data_cfg, "max_seq_len").text = str(
        config.data.max_seq_len
    )
    ET.SubElement(data_cfg, "min_freq").text = str(
        config.data.min_freq
    )
    ET.SubElement(
        data_cfg, "train_samples_per_author"
    ).text = str(config.data.train_samples_per_author)
    ET.SubElement(
        data_cfg, "val_samples_per_author"
    ).text = str(config.data.val_samples_per_author)
    ET.SubElement(
        data_cfg, "divider_target_count"
    ).text = str(config.data.divider_target_count)

    # --- per-epoch metrics ---
    epochs_el = ET.SubElement(root, "epochs")
    num_epochs = len(history["loss"])
    for i in range(num_epochs):
        ep = ET.SubElement(
            epochs_el, "epoch", number=str(i + 1)
        )
        ET.SubElement(
            ep, "train_loss"
        ).text = f"{history['loss'][i]:.6f}"
        ET.SubElement(
            ep, "val_loss"
        ).text = f"{history['val_loss'][i]:.6f}"
        ET.SubElement(
            ep, "val_acc"
        ).text = f"{history['val_acc'][i]:.4f}"
        ET.SubElement(
            ep, "val_f1"
        ).text = f"{history['val_f1'][i]:.4f}"
        ET.SubElement(
            ep, "val_precision"
        ).text = f"{history['val_precision'][i]:.4f}"
        ET.SubElement(
            ep, "val_recall"
        ).text = f"{history['val_recall'][i]:.4f}"
        ET.SubElement(
            ep, "val_auc_roc"
        ).text = f"{history['val_auc_roc'][i]:.4f}"
        ET.SubElement(
            ep, "lr"
        ).text = f"{history['lr'][i]:.8f}"
        ET.SubElement(
            ep, "avg_grad_norm"
        ).text = f"{history['avg_grad_norm'][i]:.4f}"

    return root


def _write_xml_report(
    history: dict[str, list[float]],
    total_params: int,
    trainable_params: int,
    config: Config,
    elapsed_seconds: float,
    status: str,
) -> None:
    """Build and flush the XML report to disk."""
    report_root = _build_xml_report(
        history=history,
        total_params=total_params,
        trainable_params=trainable_params,
        config=config,
        training_duration_seconds=elapsed_seconds,
        status=status,
    )
    _indent_xml(report_root)
    ET.ElementTree(report_root).write(
        str(TRAINING_REPORT),
        encoding="utf-8",
        xml_declaration=True,
    )


def _indent_xml(
    element: ET.Element, level: int = 0
) -> None:
    """Add pretty-print indentation to an ElementTree in-place (Python < 3.9 compat)."""
    indent = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        if not element.tail or not element.tail.strip():
            element.tail = indent
        for child in element:
            _indent_xml(child, level + 1)
        # last child tail
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (
            not element.tail or not element.tail.strip()
        ):
            element.tail = indent


def train_model(config: Config) -> None:
    # 1. DATA LOADING
    logger.info("--- 1. Loading Data ---")
    divider: Divider = Divider(
        target_count=config.data.divider_target_count,
        min_length=config.data.max_seq_len,
    )

    raw_data: dict[Any, list[str]] = (
        divider.parse_directory(
            str(constants.SAVE_FOLDER_UKRLIB)
        )
    )

    author_ids: list[Any] = list(raw_data.keys())
    logger.info("Total authors loaded: %d", len(author_ids))

    # 2. Train (80%) / Val (20%) split
    train_data_dict, val_data_dict = split_author_chunks(
        raw_data
    )

    logger.info(
        "Authors in Train: %d | Authors in Val: %d",
        len(train_data_dict),
        len(val_data_dict),
    )

    # 3. Vocabulary (cache-aware)
    all_train_texts: list[str] = [
        t
        for texts in train_data_dict.values()
        for t in texts
    ]

    fingerprint = compute_data_fingerprint(
        constants.SAVE_FOLDER_UKRLIB
    )
    cache_valid = (
        not config.data.vocab_force_rebuild
        and Vocabulary.is_cache_valid(
            VOCAB_CACHE, fingerprint
        )
    )

    if cache_valid:
        logger.info(
            "Vocabulary cache is up-to-date — loading from %s",
            VOCAB_CACHE,
        )
        vocab: Vocabulary = Vocabulary.load(
            VOCAB_CACHE, config
        )
    else:
        if config.data.vocab_force_rebuild:
            logger.info(
                "VOCAB_FORCE_REBUILD=True — rebuilding vocabulary..."
            )
        else:
            logger.info(
                "Data changed or cache missing — rebuilding vocabulary..."
            )
        vocab = Vocabulary(config)
        vocab.build(all_train_texts)
        vocab.save(VOCAB_CACHE, fingerprint=fingerprint)

    # 4. Pair generation
    logger.info("--- Generating Pairs ---")
    train_pairs, train_labels, val_pairs, val_labels = (
        generate_split_pairs(
            train_data_dict,
            val_data_dict,
            config.data.train_samples_per_author,
            config.data.val_samples_per_author,
        )
    )

    logger.info(
        "Train pairs: %d | Val pairs: %d",
        len(train_pairs),
        len(val_pairs),
    )

    train_loader: DataLoader = DataLoader(
        AuthorPairDataset(train_pairs, train_labels, vocab),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader: DataLoader = DataLoader(
        AuthorPairDataset(val_pairs, val_labels, vocab),
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    # 5. Model
    model: SiameseLSTM = SiameseLSTM(
        vocab.vocab_size,
        config.model.embed_dim,
        config.model.hidden_dim,
        config.model.classifier_h1,
        config.model.classifier_h2,
        config.model.dropout,
        config.model.embed_dropout,
    ).to(config.device)

    # Parameter summary — logged once before training starts
    _log_model_summary(model)
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    optimizer: optim.AdamW = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=config.training.scheduler_min_lr,
    )

    # 6. Training loop
    history: dict[str, list[float]] = {
        "loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_auc_roc": [],
        "lr": [],
        "avg_grad_norm": [],
    }

    logger.info("--- Starting Training ---")
    best_val_loss: float = float("inf")
    training_start = time.monotonic()

    for epoch in range(config.training.epochs):
        model.train()
        total_loss: float = 0.0
        batch_grad_norms: list[float] = []

        for t1, t2, label in train_loader:
            t1 = t1.to(config.device)
            t2 = t2.to(config.device)
            label = label.to(config.device)

            optimizer.zero_grad()
            output: torch.Tensor = model(t1, t2)
            loss: torch.Tensor = criterion(
                output, label.unsqueeze(1)
            )
            loss.backward()

            # Compute global gradient norm BEFORE clipping
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += (
                        p.grad.detach().data.norm(2).item()
                        ** 2
                    )
            grad_norm = grad_norm**0.5
            batch_grad_norms.append(grad_norm)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.training.grad_clip_max_norm,
            )
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss: float = total_loss / len(
            train_loader
        )
        avg_grad_norm: float = (
            sum(batch_grad_norms) / len(batch_grad_norms)
            if batch_grad_norms
            else 0.0
        )
        current_lr: float = _get_current_lr(optimizer)

        # Validation
        model.eval()
        val_loss: float = 0.0
        correct: float = 0.0
        total: int = 0
        all_preds: list[int] = []
        all_probs: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for t1, t2, label in val_loader:
                t1 = t1.to(config.device)
                t2 = t2.to(config.device)
                label = label.to(config.device)

                output = model(t1, t2)
                loss = criterion(output, label.unsqueeze(1))
                val_loss += loss.item()

                probs: torch.Tensor = torch.sigmoid(
                    output
                ).squeeze()
                preds: torch.Tensor = (probs > 0.5).float()
                correct += (
                    (preds.squeeze() == label).sum().item()
                )
                total += label.size(0)

                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().int().tolist())
                all_labels.extend(
                    label.cpu().int().tolist()
                )

        avg_val_loss: float = val_loss / len(val_loader)
        val_acc: float = 100.0 * correct / total

        # Extended validation metrics (sklearn)
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", UndefinedMetricWarning
            )
            val_f1: float = float(
                f1_score(all_labels, all_preds)
            )
            val_precision: float = float(
                precision_score(all_labels, all_preds)
            )
            val_recall: float = float(
                recall_score(all_labels, all_preds)
            )
        try:
            val_auc: float = float(
                roc_auc_score(all_labels, all_probs)
            )
        except ValueError:
            # Can happen when only one class is present in a tiny val set
            val_auc = 0.0
            logger.warning(
                "AUC-ROC could not be computed for epoch %d (single class in val set).",
                epoch + 1,
            )

        history["loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_auc_roc"].append(val_auc)
        history["lr"].append(current_lr)
        history["avg_grad_norm"].append(avg_grad_norm)

        scheduler.step(avg_val_loss)

        logger.info(
            "Epoch %d/%d | Loss: %.4f | Val Loss: %.4f | Acc: %.2f%% "
            "| F1: %.4f | Prec: %.4f | Rec: %.4f | AUC: %.4f "
            "| LR: %.2e | GradNorm: %.4f",
            epoch + 1,
            config.training.epochs,
            avg_train_loss,
            avg_val_loss,
            val_acc,
            val_f1,
            val_precision,
            val_recall,
            val_auc,
            current_lr,
            avg_grad_norm,
        )

        # Write XML report after every epoch so it survives interruptions
        _write_xml_report(
            history=history,
            total_params=total_params,
            trainable_params=trainable_params,
            config=config,
            elapsed_seconds=time.monotonic()
            - training_start,
            status="in_progress",
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint: dict[str, Any] = {
                "model_state": model.state_dict(),
                "vocab_word2idx": vocab.word2idx,
                "config": {
                    "max_seq_len": config.data.max_seq_len,
                    "embed_dim": config.model.embed_dim,
                    "hidden_dim": config.model.hidden_dim,
                    "classifier_h1": config.model.classifier_h1,
                    "classifier_h2": config.model.classifier_h2,
                    "dropout": config.model.dropout,
                    "seed": config.seed,
                },
            }
            torch.save(checkpoint, CHECKPOINT)
            logger.info(
                "Checkpoint saved (val_loss=%.4f) → %s",
                best_val_loss,
                CHECKPOINT,
            )

    training_duration = time.monotonic() - training_start

    logger.info(
        "Training finished. Best val_loss: %.4f | Duration: %.1fs",
        best_val_loss,
        training_duration,
    )

    # 7. Write final XML report with status=completed
    _write_xml_report(
        history=history,
        total_params=total_params,
        trainable_params=trainable_params,
        config=config,
        elapsed_seconds=training_duration,
        status="completed",
    )
    logger.info(
        "Training report saved → %s", TRAINING_REPORT
    )
