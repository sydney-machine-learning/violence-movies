"""Clip models accepting the project's existing BGR 112x112 memmap batches.

TorchVision is imported only when constructing MViTv2-S. Input adaptation is
inside the wrapper, so training, inference and timing use the same source data.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class MViTInputAdapter(nn.Module):
    """BCTHW BGR pixels (0..255) -> RGB normalized BCTHW, 112 -> 224.

Direct bilinear upsampling preserves the entire prepared frame. This deliberately
does not use the pretrained recipe's resize-to-256 and center-crop operation.
"""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.full((1, 3, 1, 1), 0.45))
        self.register_buffer("std", torch.full((1, 3, 1, 1), 0.225))

    @staticmethod
    def configuration():
        return {
            "version": 1,
            "source_layout": "BCTHW", "source_color": "BGR",
            "source_frames": 16, "source_size": [112, 112],
            "source_range": [0, 255], "network_color": "RGB",
            "network_size": [224, 224], "scale_divisor": 255.0,
            "resize": "bilinear", "align_corners": False, "antialias": False,
            "crop": None, "mean": [0.45] * 3, "std": [0.225] * 3,
        }

    def forward(self, inputs):
        if inputs.ndim != 5 or tuple(inputs.shape[1:]) != (3, 16, 112, 112):
            raise ValueError("MViTv2-S expects source BGR batches [B, 3, 16, 112, 112]")
        if not inputs.is_floating_point() and inputs.dtype != torch.uint8:
            raise TypeError("Expected floating pixel values in 0..255 or uint8 pixels")
        batch = inputs.shape[0]
        dtype = inputs.dtype if inputs.is_floating_point() else torch.float32
        frames = inputs[:, [2, 1, 0]].permute(0, 2, 1, 3, 4)
        frames = frames.reshape(-1, 3, 112, 112).to(dtype=dtype) / 255.0
        frames = F.interpolate(frames, size=(224, 224), mode="bilinear",
                               align_corners=False, antialias=False)
        frames = (frames - self.mean.to(dtype=dtype)) / self.std.to(dtype=dtype)
        return frames.reshape(batch, 16, 3, 224, 224).permute(0, 2, 1, 3, 4).contiguous()


class MViTv2SClassifier(nn.Module):
    """MViTv2-S with an internal adapter and a task-specific output head."""

    def __init__(self, n_classes=3, weights="KINETICS400_V1", freeze_backbone=False):
        super().__init__()
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        if n_classes < 1:
            raise ValueError("n_classes must be positive")
        resolved_weights = MViT_V2_S_Weights.verify(weights)
        self.pretrained_weights = resolved_weights.name if resolved_weights is not None else None
        self.n_classes = n_classes
        self.freeze_backbone = bool(freeze_backbone)
        self.input_adapter = MViTInputAdapter()
        # Load the original pretrained head before replacing it with our head.
        self.backbone = mvit_v2_s(weights=resolved_weights)
        in_features = self.backbone.head[-1].in_features
        self.backbone.head[-1] = nn.Linear(in_features, n_classes)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self.freeze_backbone)
        for parameter in self.backbone.head.parameters():
            parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # A linear probe must also disable backbone stochastic depth/dropout.
            self.backbone.eval()
            self.backbone.head.train(mode)
        return self

    def forward(self, inputs):
        return self.backbone(self.input_adapter(inputs))

    def checkpoint_metadata(self):
        return {
            "model_name": "mvit_v2_s", "n_classes": self.n_classes,
            "pretrained_weights": self.pretrained_weights,
            "freeze_backbone": self.freeze_backbone,
            "input_adapter": self.input_adapter.configuration(),
            "torch_version": str(torch.__version__),
            "torchvision_version": version("torchvision"),
        }


def getMViTv2SModel(verbose=True, freeze_backbone=False, lr=1e-5,
                   weights="KINETICS400_V1", n_classes=3):
    """Training factory; lr is accepted for compatibility with existing factories.

    Set weights=None only for checkpoint restoration or explicit untrained tests.
    Fine-tuning/linear probing is selected explicitly with freeze_backbone.
    """
    model = MViTv2SClassifier(n_classes=n_classes, weights=weights,
                              freeze_backbone=freeze_backbone)
    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"MViTv2SClassifier: {trainable:,} trainable / {total:,} total parameters")
        print(f"Pretraining: {model.pretrained_weights}; source BGR 16x112x112 -> RGB 16x224x224")
    return model


def mvit_active_parameter_names(model):
    """The standard MViTv2-S forward uses every registered parameter, frozen too."""
    if not isinstance(model, MViTv2SClassifier):
        raise TypeError("Expected MViTv2SClassifier")
    return [name for name, _ in model.named_parameters()]


def mvit_benchmark_case(project_root, batch_size=1, checkpoint=None):
    """Time adaptation plus transformer on the same 112x112 source workload."""
    from model_benchmark import BenchmarkCase, Factory

    return BenchmarkCase(
        name="MViTv2-S",
        model_factory=Factory("video_models:getMViTv2SModel", {
            "weights": None, "freeze_backbone": False, "verbose": False,
        }),
        checkpoint=str(checkpoint or Path(project_root) / "weights" / "MViTv2S.pt"),
        input_shape=(batch_size, 3, 16, 112, 112), batch_size=batch_size,
        input_description="BCTHW; BGR 0..255; 112x112 source -> 224x224 RGB; adapter inside timed forward",
        parameter_selector=Factory("video_models:mvit_active_parameter_names"),
        mac_counter=Factory("model_benchmark:count_forward_macs"),
        metadata={"model_name": "mvit_v2_s", "source_size": [112, 112],
                  "network_size": [224, 224], "adapter_in_timed_forward": True,
                  "input_adapter": MViTInputAdapter.configuration()},
    )
