"""
Evaluation Metrics for Semantic Segmentation.

Includes: mIoU, per-class IoU, pixel accuracy, F1 score.

面试抓手:
  Pixel Acc 容易被背景像素占比“抬高”；mIoU 和 mean F1 更能反映单层/少层等少数类效果。
"""

import numpy as np
import torch


class SegmentationMetrics:
    """Compute semantic segmentation metrics.
    
    Maintains a confusion matrix and computes metrics from it.

    confusion_matrix[row, col] 表示真实类别 row 被预测成 col 的像素数量。
    所有指标都可以从这张表推出来，便于统一统计 train/val/test。
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, preds, targets):
        """Update confusion matrix with batch predictions.
        
        Args:
            preds: [B, H, W] predicted class indices
            targets: [B, H, W] ground truth class indices
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        mask = (targets >= 0) & (targets < self.num_classes)
        label = self.num_classes * targets[mask].astype(int) + preds[mask].astype(int)
        count = np.bincount(label, minlength=self.num_classes ** 2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)

    def get_iou(self):
        """Compute per-class IoU and mIoU."""
        intersection = np.diag(self.confusion_matrix)
        # IoU = TP / (TP + FP + FN)，语义分割里比整体像素准确率更关键。
        union = (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            intersection
        )
        iou = intersection / (union + 1e-10)
        return iou

    def get_miou(self):
        """Compute mean IoU."""
        return self.get_iou().mean()

    def get_pixel_accuracy(self):
        """Compute overall pixel accuracy."""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / (total + 1e-10)

    def get_class_accuracy(self):
        """Compute per-class accuracy."""
        correct = np.diag(self.confusion_matrix)
        total = self.confusion_matrix.sum(axis=1)
        return correct / (total + 1e-10)

    def get_f1(self):
        """Compute per-class F1 score."""
        precision = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=0) + 1e-10
        )
        recall = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=1) + 1e-10
        )
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return f1

    def get_results(self):
        """Get all metrics as a dictionary."""
        iou = self.get_iou()
        return {
            'mIoU': self.get_miou(),
            'per_class_iou': iou,
            'pixel_acc': self.get_pixel_accuracy(),
            'class_acc': self.get_class_accuracy(),
            'f1': self.get_f1(),
            'mean_f1': self.get_f1().mean(),
        }
