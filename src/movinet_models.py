"""Pinned MoViNet A0: Base for chunks, Stream for complete source videos.

All adapters start from the existing 112x112 BGR pixels. No extra packages,
source-video decode or prepared-array changes are needed. Native cached states
are used only for segmented inference; training is one complete causal forward.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F


UPSTREAM_COMMIT = "c2d1edf48fc6c5259707f9d833f22171b4f63493"
STREAM_NAME = "movinet_a0_stream"
BASE_NAME = "movinet_a0_base"
STREAM_LABEL = "MoViNet-A0-Stream"
MOVINET_SCHEMA = 1
WEIGHTS = {
    "base": ("modelA0_statedict_v3", "7fa4d3b5a1c3970d581476782f8892c24d46c10433b7bd8ea4fc4ccdc36224e1"),
    "stream": ("modelA0_stream_statedict_v3", "447c0554daa6bebdcf6fc69b2651b25b29cc69e003da4e6ff56f9a2488f403cf"),
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pretrained_checkpoint(variant, cache_dir=None):
    name, expected = WEIGHTS[variant]
    folder = (Path(cache_dir) if cache_dir else
              Path(__file__).resolve().parents[1] / "results/pretrained_cache/movinet")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"a0_{variant}.pt"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/Atze00/MoViNet-pytorch/{UPSTREAM_COMMIT}/weights/{name}"
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.hub.download_url_to_file(url, str(temporary), progress=True)
            if _sha256(temporary) != expected:
                raise ValueError("Downloaded MoViNet checkpoint failed SHA-256 verification")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if _sha256(path) != expected:
        raise ValueError(f"MoViNet checkpoint hash mismatch; inspect {path}")
    return path


class MoViNetInputAdapter(nn.Module):
    """BCTHW BGR 0..255 -> RGB 0..1, direct full-frame 112 -> 172 resize.

    Matches the port's zero-to-one recipe; no ImageNet mean/std normalization.
    Temporal length is preserved, including 8-frame continuation updates.
    """

    @staticmethod
    def configuration():
        return {"source_layout": "BCTHW", "source_color": "BGR", "source_size": [112, 112],
                "source_range": [0, 255], "network_color": "RGB", "network_size": [172, 172],
                "scale_divisor": 255., "resize": "bilinear", "align_corners": False,
                "antialias": False, "crop": None, "mean": None, "std": None}

    def forward(self, inputs):
        if inputs.ndim != 5 or inputs.shape[1] != 3 or tuple(inputs.shape[-2:]) != (112, 112) or inputs.shape[2] < 1:
            raise ValueError("MoViNet expects BGR [B,3,T,112,112] with T >= 1")
        if not inputs.is_floating_point() and inputs.dtype != torch.uint8:
            raise TypeError("Expected floating or uint8 BGR pixels")
        batch, _, frames, _, _ = inputs.shape
        dtype = inputs.dtype if inputs.is_floating_point() else torch.float32
        images = inputs[:, [2, 1, 0]].permute(0, 2, 1, 3, 4).reshape(-1, 3, 112, 112)
        images = images.to(dtype=dtype) / 255.
        images = F.interpolate(images, (172, 172), mode="bilinear", align_corners=False, antialias=False)
        return images.reshape(batch, frames, 3, 172, 172).permute(0, 2, 1, 3, 4).contiguous()


def prepare_unique_video(chunks, record):
    """Use frame identities, not pixel equality, to remove overlap and padding.

    Returns observed BGR frames [C,T,H,W] and a cumulative length per source
    window. Repeated padding contributes no new observation.
    """
    rows = record["rows"]
    if chunks.ndim != 5 or chunks.shape[1] != 3 or chunks.shape[2] != 16 or len(rows) != len(chunks):
        raise ValueError("Expected existing [K,3,16,H,W] chunks with one metadata row per chunk")
    selected, ends, last, total = [], [], -1, 0
    previous_start = None
    for index, row in enumerate(rows):
        indices = row["sampled_frame_indices"]
        if (len(indices) != 16 or row["start_frame"] != indices[0]
                or any(b < a for a, b in zip(indices, indices[1:]))
                or indices[0] < 0 or (previous_start is not None and indices[0] <= previous_start)):
            raise ValueError("Invalid frame order for streaming reconstruction")
        if previous_start is not None and row["start_frame"] - previous_start != 24:
            raise ValueError("Discontinuous source windows require a state reset")
        keep = []
        for position, frame in enumerate(indices):
            if frame > last:
                keep.append(position)
                last = frame
        if keep:
            selected.append(chunks[index, :, keep])
            total += len(keep)
        ends.append(total)
        previous_start = indices[0]
    if not selected:
        raise ValueError("No observed frames in source video")
    return {"frames": torch.cat(selected, dim=1), "prefix_ends": tuple(ends)}


class MoViNetA0Classifier(nn.Module):
    def __init__(self, variant="base", pretrained=True, freeze_backbone=False,
                 freeze_bn_stats=True, n_classes=3, cache_dir=None):
        super().__init__()
        from _vendor.movinet_a0.models import MoViNet
        from _vendor.movinet_a0.config import _C

        if variant not in WEIGHTS or n_classes < 1:
            raise ValueError("Expected A0 base/stream and positive class count")
        self.variant, self.n_classes = variant, n_classes
        self.model_name = BASE_NAME if variant == "base" else STREAM_NAME
        self.carry_state = variant == "stream"
        self.freeze_backbone, self.freeze_bn_stats = bool(freeze_backbone), bool(freeze_bn_stats)
        self.input_adapter = MoViNetInputAdapter()
        # Explicit architecture is necessary even when restoring with no download.
        self.backbone = MoViNet(_C.MODEL.MoViNetA0, causal=variant == "stream",
                               pretrained=False, num_classes=600, tf_like=True,
                               conv_type="3d" if variant == "base" else "2plus1d")
        self.pretrained_weights = f"KINETICS600_{variant.upper()}_PORT_V3" if pretrained else None
        if pretrained:
            state = torch.load(pretrained_checkpoint(variant, cache_dir), map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
            del state
        # Upstream tutorial's task head: load all 600-way weights, then replace.
        self.backbone.classifier[3] = nn.Conv3d(2048, n_classes, (1, 1, 1))
        self._freeze_policy()
        self.train(self.training)

    def _freeze_policy(self):
        self.backbone.requires_grad_(not self.freeze_backbone)
        self.backbone.classifier.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
            self.backbone.classifier.train(mode)
        if self.freeze_bn_stats:
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
        return self

    def reset_stream(self):
        self.backbone.clean_activation_buffers()

    def _independent_forward(self, frames):
        self.reset_stream()
        try:
            return self.backbone(self.input_adapter(frames))
        finally:
            # Clearing detached attributes is safe before backward and prevents
            # inference tensors/counters leaking into another video or phase.
            self.reset_stream()

    def forward(self, chunks):
        if self.variant == "base" and (chunks.ndim != 5 or chunks.shape[2] != 16):
            raise ValueError("A0-Base chunk workflow expects exactly 16 sampled frames")
        return self._independent_forward(chunks)

    def prepare_video(self, chunks, record):
        if self.variant != "stream":
            raise ValueError("Base belongs to the independent-chunk workflow")
        return prepare_unique_video(chunks, record)

    @staticmethod
    def _video_parts(video, length=None):
        if isinstance(video, dict):
            frames, ends = video["frames"], tuple(video["prefix_ends"])
        elif isinstance(video, torch.Tensor) and video.ndim == 4:
            frames, ends = video, (video.shape[1],)
        else:
            raise ValueError("Pass prepared unique frames or call prepare_video(chunks, record)")
        count = len(ends) if length is None else int(length)
        if not 1 <= count <= len(ends) or not ends or ends[-1] > frames.shape[1]:
            raise ValueError("Invalid valid video length")
        if any(b < a for a, b in zip((0, *ends[:-1]), ends)) or ends[0] < 1:
            raise ValueError("Invalid prefix lengths")
        return frames[:, :ends[count - 1]], ends[:count]

    def forward_video(self, video, length=None):
        if self.variant != "stream":
            raise ValueError("Use the chunk forward for Base")
        frames, _ = self._video_parts(video, length)
        device = next(self.parameters()).device
        return self._independent_forward(frames[None].to(device)), None

    def stream_step(self, frames):
        if self.variant != "stream":
            raise ValueError("State carry requires the Stream architecture")
        if self.training and torch.is_grad_enabled():
            raise RuntimeError("Use forward_video for training; native between-call buffers detach gradients")
        return self.backbone(self.input_adapter(frames))

    def forward_sequence(self, video, length=None):
        """Evaluation-only prefix outputs on new observations; reset at both ends."""
        frames, ends = self._video_parts(video, length)
        device = next(self.parameters()).device
        outputs, start = [], 0
        self.reset_stream()
        try:
            for end in ends:
                if end > start:
                    outputs.append(self.stream_step(frames[:, start:end][None].to(device)))
                else:
                    outputs.append(outputs[-1])
                start = end
            return torch.cat(outputs, 0), None
        finally:
            self.reset_stream()

    def checkpoint_metadata(self):
        return {"schema_version": MOVINET_SCHEMA, "model_name": self.model_name,
                "variant": self.variant, "n_classes": self.n_classes,
                "upstream_commit": UPSTREAM_COMMIT, "pretrained_weights": self.pretrained_weights,
                "pretrained_sha256": WEIGHTS[self.variant][1] if self.pretrained_weights else None,
                "freeze_backbone": self.freeze_backbone, "freeze_bn_stats": self.freeze_bn_stats,
                "input_adapter": self.input_adapter.configuration(),
                "video_readout": "final native cumulative output" if self.carry_state else "independent chunk logits",
                "training": "one complete causal forward" if self.carry_state else "independent 16-frame chunks",
                "frame_policy": "metadata: observed frames once, no repeated padding" if self.carry_state else "existing prepared chunks"}


def getMoViNetA0BaseModel(verbose=True, freeze_backbone=False, lr=1e-5,
                        weights="KINETICS600", freeze_bn_stats=True, n_classes=3):
    if weights not in (None, "KINETICS600"):
        raise ValueError("Expected KINETICS600 or None for project-checkpoint restoration")
    return _describe(MoViNetA0Classifier("base", weights is not None, freeze_backbone,
                                        freeze_bn_stats, n_classes), verbose)


def build_movinet_stream(pretrained=True, verbose=True, freeze_backbone=False,
                         freeze_bn_stats=True, n_classes=3):
    return _describe(MoViNetA0Classifier("stream", pretrained, freeze_backbone,
                                        freeze_bn_stats, n_classes), verbose)


def _describe(model, verbose):
    if verbose:
        print(f"{model.model_name}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
              f"trainable / {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def load_movinet_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("model_metadata", {})
    expected = model.checkpoint_metadata()
    for key in ("schema_version", "model_name", "variant", "n_classes", "upstream_commit",
                "input_adapter", "video_readout", "training", "frame_policy"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"MoViNet checkpoint policy mismatch: {key}")
    pretrained_id = f"KINETICS600_{model.variant.upper()}_PORT_V3"
    if (metadata.get("pretrained_weights") not in (None, pretrained_id)
            or metadata.get("pretrained_sha256") != (WEIGHTS[model.variant][1] if metadata.get("pretrained_weights") else None)):
        raise ValueError("MoViNet checkpoint pretraining provenance mismatch")
    from cnn_lstm_pytorch import CLASS_NAMES
    if payload.get("class_names") != CLASS_NAMES:
        raise ValueError("MoViNet checkpoint class ordering mismatch")
    model.reset_stream()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.freeze_backbone = bool(metadata["freeze_backbone"])
    model.freeze_bn_stats = bool(metadata["freeze_bn_stats"])
    model.pretrained_weights = metadata["pretrained_weights"]
    model._freeze_policy()
    model.train(model.training)
    return {"model_class": payload.get("model_class"), "class_names": payload["class_names"],
            "epoch": payload.get("epoch"), "model_metadata": metadata,
            "training_config": payload.get("training_config", {})}


def movinet_parameter_names(model):
    if not isinstance(model, MoViNetA0Classifier):
        raise TypeError("Expected pinned MoViNet A0")
    return [name for name, _ in model.named_parameters()]


def count_movinet_macs(model, adapter, inputs):
    from model_benchmark import count_conv_linear_macs
    movinet_parameter_names(model)
    result = count_conv_linear_macs(model, adapter, inputs)
    result["scope"] = ("Dense Conv2d/Conv3d/Linear MACs on every native call, including depthwise and SE projections. "
                       "Excludes bias, normalization, pooling/cumulative sums, elementwise operations, activations and interpolation.")
    return result


class MoViNetStreamAdapter:
    def __init__(self, model):
        self.model = model

    def reset(self):
        self.model.reset_stream()

    def step(self, inputs):
        return self.model.stream_step(inputs)


class MoViNetVideoAdapter(MoViNetStreamAdapter):
    def __init__(self, model, chunks=5):
        super().__init__(model)
        self.ends = tuple(16 + 8 * i for i in range(chunks))

    def step(self, inputs):
        return torch.cat([self.model.forward_sequence({"frames": video, "prefix_ends": self.ends})[0][-1:]
                          for video in inputs], 0)


def movinet_base_benchmark_case(project_root, batch_size=1, checkpoint=None):
    from model_benchmark import BenchmarkCase, Factory
    return BenchmarkCase(
        name="MoViNet-A0-Base", model_factory=Factory("movinet_models:getMoViNetA0BaseModel", {
            "weights": None, "freeze_backbone": False, "verbose": False}),
        checkpoint=str(checkpoint or Path(project_root) / "weights/MoViNetA0Base.pt"),
        checkpoint_loader=Factory("movinet_models:load_movinet_checkpoint"),
        input_shape=(batch_size, 3, 16, 112, 112), batch_size=batch_size,
        parameter_selector=Factory("movinet_models:movinet_parameter_names"),
        mac_counter=Factory("movinet_models:count_movinet_macs"),
        input_description="16-frame BGR source 112x112 -> internal RGB 0..1 172x172",
        metadata={"model_name": BASE_NAME, "source_size": [112, 112], "network_size": [172, 172],
                  "adapter_in_timed_forward": True, "decision": "existing two_step outside timed forward"})


def movinet_stream_benchmark_cases(project_root, experiment_tag="airtlab_movinet_stream_v1",
                                   batch_size=1, video_chunks=5, workloads=("cold", "continuation", "video")):
    from model_benchmark import BenchmarkCase, Factory
    if video_chunks < 1:
        raise ValueError("video_chunks must be positive")
    cases = []
    for workload in workloads:
        if workload not in {"cold", "continuation", "video"}:
            raise ValueError("Unknown MoViNet workload")
        count = video_chunks if workload == "video" else 1
        frames = 16 + 8 * (count - 1) if workload == "video" else 8 if workload == "continuation" else 16
        cases.append(BenchmarkCase(
            name=f"MoViNet-A0-Stream / {workload}",
            model_factory=Factory("movinet_models:build_movinet_stream", {"pretrained": False, "verbose": False}),
            checkpoint=str(Path(project_root) / "weights/temporal" / experiment_tag / f"{STREAM_NAME}.pt"),
            checkpoint_loader=Factory("movinet_models:load_movinet_checkpoint"),
            adapter_factory=Factory("movinet_models:MoViNetVideoAdapter", {"chunks": count}) if workload == "video"
                            else Factory("movinet_models:MoViNetStreamAdapter"),
            input_shape=(batch_size, 3, frames, 112, 112), batch_size=batch_size, frames_per_call=frames,
            parameter_selector=Factory("movinet_models:movinet_parameter_names"),
            mac_counter=Factory("movinet_models:count_movinet_macs"),
            mode="stream" if workload == "continuation" else "clip",
            state_policy="carry" if workload == "continuation" else "reset_each_call",
            input_description=f"BGR 112x112 unique observations; {frames} frames; internal RGB 172x172 adapter",
            comparison_group=f"movinet_{workload}_{frames}_unique_frames_112",
            metadata={"model_name": STREAM_NAME, "workload": workload, "chunks_per_call": count,
                      "processed_chunks_per_call": count, "processed_tensor_frames_per_item": frames,
                      "model_carries_state": True, "source_size": [112, 112], "network_size": [172, 172],
                      "adapter_in_timed_forward": True, "deduplication_in_timed_forward": False,
                      "head_calls_per_item": count, "aggregation_in_timed_forward": False}))
    return cases
