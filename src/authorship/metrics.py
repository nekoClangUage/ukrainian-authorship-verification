"""Metrics accumulator for authorship model evaluation."""

from dataclasses import dataclass, field

import torch


@dataclass
class AuthorshipMetrics:
    """Accumulates batch-level loss and classification accuracy.

    Usage:
        metrics = AuthorshipMetrics()
        for t1, t2, label in val_loader:
            output = model(t1, t2)
            loss = criterion(output, label.unsqueeze(1))
            preds = (torch.sigmoid(output) > 0.5).float()
            metrics.update(loss.item(), preds, label)

        print(metrics)
        metrics.reset()
    """

    _total_loss: float = field(default=0.0, repr=False)
    _correct: int = field(default=0, repr=False)
    _total: int = field(default=0, repr=False)
    _steps: int = field(default=0, repr=False)

    def update(
        self,
        batch_loss: float,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Register the results of one batch.

        Args:
            batch_loss:   scalar value from loss.item()
            predictions:  boolean or float32 tensor of shape [B] or [B, 1]
            labels:       ground-truth tensor of shape [B]
        """
        self._total_loss += batch_loss
        self._steps += 1
        self._correct += int(
            (predictions.squeeze() == labels).sum().item()
        )
        self._total += labels.size(0)

    @property
    def accuracy(self) -> float:
        """Classification accuracy in percent (0–100)."""
        if self._total == 0:
            return 0.0
        return 100.0 * self._correct / self._total

    @property
    def average_loss(self) -> float:
        """Mean loss across all accumulated batches."""
        if self._steps == 0:
            return 0.0
        return self._total_loss / self._steps

    def reset(self) -> None:
        """Zero all counters for the next epoch."""
        self._total_loss = 0.0
        self._correct = 0
        self._total = 0
        self._steps = 0

    def __repr__(self) -> str:
        return (
            f"AuthorshipMetrics("
            f"loss={self.average_loss:.4f}, "
            f"accuracy={self.accuracy:.2f}%)"
        )
