from __future__ import annotations

import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["non-violence", "low-level violence", "high-level violence"]
FRAMES_PER_CHUNK = 16
FRAME_SAMPLE_STRIDE = 3
CHUNK_OVERLAP = 0.5
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
FRAME_CHANNELS = 3
SAMPLE_DTYPE = np.float32
LABEL_DTYPE = np.int8
VIDEO_DTYPE = np.int32
SAMPLE_SHAPE = (FRAMES_PER_CHUNK, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
RAW_FRAMES_PER_CHUNK = 1 + (FRAMES_PER_CHUNK - 1) * FRAME_SAMPLE_STRIDE
CHUNK_FRAME_STEP = max(1, int(round(FRAMES_PER_CHUNK * (1.0 - CHUNK_OVERLAP) * FRAME_SAMPLE_STRIDE)))
DEFAULT_VIOLENCE_PRESENCE_THRESHOLD = 0.60  # previously 0.70
DEFAULT_VIOLENT_CHUNK_THRESHOLD = 0.50     # previously 0.60
DEFAULT_MIN_VIOLENT_CHUNK_FRACTION = 0.30  # keep unchanged
DEFAULT_VIOLENCE_MEAN_MARGIN = 0.05        # previously 0.15
DEFAULT_HIGH_VIOLENCE_EVIDENCE_THRESHOLD = 0.70  # previously 0.80
DEFAULT_HIGH_VIOLENCE_MEAN_THRESHOLD = 0.60     # previously 0.65
DEFAULT_HIGH_VIOLENCE_MARGIN = 0.20             # previously 0.25
DEFAULT_VIDEO_MEAN_WEIGHT = 0.3
DEFAULT_VIDEO_MAX_WEIGHT = 0.7


def get_project_root() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name == "src" else cwd


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_mmap_chunk_count(
    dataset_base_path: Path | str,
    labels_name: str = "labels.mmap",
    samples_name: str = "samples.mmap",
    videos_name: str = "videos.mmap",
) -> int:
    dataset_base_path = Path(dataset_base_path)
    labels_path = dataset_base_path / labels_name
    samples_path = dataset_base_path / samples_name
    videos_path = dataset_base_path / videos_name

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels memmap: {labels_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples memmap: {samples_path}")
    if not videos_path.exists():
        raise FileNotFoundError(f"Missing videos memmap: {videos_path}")

    label_itemsize = np.dtype(LABEL_DTYPE).itemsize
    if labels_path.stat().st_size % label_itemsize != 0:
        raise ValueError(f"{labels_path} size is not divisible by {label_itemsize}")

    chunk_count = labels_path.stat().st_size // label_itemsize
    expected_sample_bytes = chunk_count * int(np.prod(SAMPLE_SHAPE)) * np.dtype(SAMPLE_DTYPE).itemsize
    expected_video_bytes = chunk_count * np.dtype(VIDEO_DTYPE).itemsize

    actual_sample_bytes = samples_path.stat().st_size
    actual_video_bytes = videos_path.stat().st_size
    if actual_sample_bytes != expected_sample_bytes:
        raise ValueError(
            f"{samples_path} has {actual_sample_bytes} bytes, expected {expected_sample_bytes} "
            f"for {chunk_count} chunks"
        )
    if actual_video_bytes != expected_video_bytes:
        raise ValueError(
            f"{videos_path} has {actual_video_bytes} bytes, expected {expected_video_bytes} "
            f"for {chunk_count} chunks"
        )

    return int(chunk_count)


def validate_mmap_dataset(
    dataset_base_path: Path | str,
    labels_name: str = "labels.mmap",
    samples_name: str = "samples.mmap",
    videos_name: str = "videos.mmap",
    class_names: list[str] | None = None,
) -> dict:
    class_names = class_names or CLASS_NAMES
    dataset_base_path = Path(dataset_base_path)
    chunk_count = infer_mmap_chunk_count(dataset_base_path, labels_name, samples_name, videos_name)

    samples = np.memmap(
        dataset_base_path / samples_name,
        mode="r",
        dtype=SAMPLE_DTYPE,
        shape=(chunk_count, *SAMPLE_SHAPE),
    )
    labels = np.memmap(dataset_base_path / labels_name, mode="r", dtype=LABEL_DTYPE, shape=(chunk_count,))
    videos = np.memmap(dataset_base_path / videos_name, mode="r", dtype=VIDEO_DTYPE, shape=(chunk_count,))

    unique_labels, label_counts = np.unique(np.asarray(labels), return_counts=True)
    label_distribution = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
    unexpected_labels = sorted(set(label_distribution) - set(range(len(class_names))))
    if unexpected_labels:
        raise ValueError(f"Unexpected labels in {dataset_base_path}: {unexpected_labels}")

    unique_video_count = int(len(np.unique(np.asarray(videos))))
    summary = {
        "dataset_base_path": str(dataset_base_path),
        "chunk_count": chunk_count,
        "sample_shape": tuple(samples.shape),
        "label_shape": tuple(labels.shape),
        "video_shape": tuple(videos.shape),
        "label_distribution": label_distribution,
        "unique_video_count": unique_video_count,
        "video_id_min": int(videos.min()) if chunk_count else None,
        "video_id_max": int(videos.max()) if chunk_count else None,
        "first_sample_min": float(samples[0].min()) if chunk_count else None,
        "first_sample_max": float(samples[0].max()) if chunk_count else None,
    }

    print("Memmap validation passed")
    print(f"  dataset: {summary['dataset_base_path']}")
    print(f"  chunks: {summary['chunk_count']}")
    print(f"  samples shape: {summary['sample_shape']}")
    print(f"  labels shape: {summary['label_shape']}")
    print(f"  videos shape: {summary['video_shape']}")
    print(f"  label distribution: {summary['label_distribution']}")
    print(f"  unique video IDs: {summary['unique_video_count']}")
    print(f"  video ID range: {summary['video_id_min']}..{summary['video_id_max']}")
    print(f"  first sample min/max: {summary['first_sample_min']:.1f}/{summary['first_sample_max']:.1f}")
    return summary


def count_chunks(video_base_path: Path | str) -> int:
    video_base_path = Path(video_base_path)
    if (
        (video_base_path / "samples.mmap").exists()
        and (video_base_path / "labels.mmap").exists()
        and (video_base_path / "videos.mmap").exists()
    ):
        return infer_mmap_chunk_count(video_base_path)

    folders = ["non-violence", "low-level violence", "high-level violence"]
    cams = ["cam1", "cam2"]
    count = 0

    for folder in folders:
        for cam_name in cams:
            path = video_base_path / folder / cam_name
            if not path.exists():
                continue

            for video_file in path.iterdir():
                if not video_file.is_file():
                    continue

                video = cv2.VideoCapture(str(video_file))
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                video.release()
                count += frame_count // 16

    return count


def combine_mmap_datasets(
    source_paths: Iterable[Path | str],
    output_path: Path | str,
    rebuild: bool = False,
    rows_per_copy: int = 256,
) -> dict:
    source_paths = [Path(path) for path in source_paths]
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    combined_files = [output_path / "samples.mmap", output_path / "labels.mmap", output_path / "videos.mmap"]
    if not rebuild and all(path.exists() for path in combined_files):
        print(f"Existing combined memmaps found at {output_path}; validating without rebuilding.")
        return validate_mmap_dataset(output_path)

    source_summaries = []
    total_chunks = 0
    for source_path in source_paths:
        summary = validate_mmap_dataset(source_path)
        source_summaries.append(summary)
        total_chunks += int(summary["chunk_count"])

    samples_out = np.memmap(
        output_path / "samples.mmap",
        mode="w+",
        dtype=SAMPLE_DTYPE,
        shape=(total_chunks, *SAMPLE_SHAPE),
    )
    labels_out = np.memmap(output_path / "labels.mmap", mode="w+", dtype=LABEL_DTYPE, shape=(total_chunks,))
    videos_out = np.memmap(output_path / "videos.mmap", mode="w+", dtype=VIDEO_DTYPE, shape=(total_chunks,))

    output_start = 0
    next_video_offset = 0
    manifest_sources = []
    for source_path, source_summary in zip(source_paths, source_summaries):
        chunk_count = int(source_summary["chunk_count"])
        samples_in = np.memmap(
            source_path / "samples.mmap",
            mode="r",
            dtype=SAMPLE_DTYPE,
            shape=(chunk_count, *SAMPLE_SHAPE),
        )
        labels_in = np.memmap(source_path / "labels.mmap", mode="r", dtype=LABEL_DTYPE, shape=(chunk_count,))
        videos_in = np.memmap(source_path / "videos.mmap", mode="r", dtype=VIDEO_DTYPE, shape=(chunk_count,))

        labels_out[output_start : output_start + chunk_count] = labels_in[:]
        videos_out[output_start : output_start + chunk_count] = videos_in[:] + next_video_offset

        for input_start in range(0, chunk_count, rows_per_copy):
            input_end = min(input_start + rows_per_copy, chunk_count)
            dest_start = output_start + input_start
            dest_end = output_start + input_end
            samples_out[dest_start:dest_end] = samples_in[input_start:input_end]

        source_video_span = int(source_summary["video_id_max"]) + 1
        manifest_sources.append(
            {
                "path": str(source_path),
                "chunk_count": chunk_count,
                "label_distribution": source_summary["label_distribution"],
                "source_video_id_min": source_summary["video_id_min"],
                "source_video_id_max": source_summary["video_id_max"],
                "combined_video_offset": next_video_offset,
                "unique_video_count": int(source_summary["unique_video_count"]),
                "video_id_span": source_video_span,
            }
        )
        output_start += chunk_count
        next_video_offset += source_video_span
        samples_out.flush()
        labels_out.flush()
        videos_out.flush()
        print(f"Combined {source_path}: wrote {output_start}/{total_chunks} chunks", flush=True)

    del samples_out, labels_out, videos_out

    manifest = {
        "class_names": CLASS_NAMES,
        "sample_shape": list(SAMPLE_SHAPE),
        "sample_dtype": np.dtype(SAMPLE_DTYPE).name,
        "label_dtype": np.dtype(LABEL_DTYPE).name,
        "video_dtype": np.dtype(VIDEO_DTYPE).name,
        "total_chunks": total_chunks,
        "total_unique_videos": sum(int(summary["unique_video_count"]) for summary in source_summaries),
        "total_video_id_span": next_video_offset,
        "sources": manifest_sources,
    }
    (output_path / "combined_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return validate_mmap_dataset(output_path)


class MMapVideoChunkDataset(Dataset):
    def __init__(
        self,
        dataset_base_path: Path | str,
        samples_name: str = "samples.mmap",
        labels_name: str = "labels.mmap",
        videos_name: str = "videos.mmap",
        indices: Iterable[int] | None = None,
        chunk_count: int | None = None,
    ):
        self.dataset_base_path = Path(dataset_base_path)
        self.chunk_count = chunk_count or infer_mmap_chunk_count(
            self.dataset_base_path,
            labels_name=labels_name,
            samples_name=samples_name,
            videos_name=videos_name,
        )
        self.samples = np.memmap(
            self.dataset_base_path / samples_name,
            mode="r",
            dtype=SAMPLE_DTYPE,
            shape=(self.chunk_count, *SAMPLE_SHAPE),
        )
        self.labels = np.memmap(
            self.dataset_base_path / labels_name,
            mode="r",
            dtype=LABEL_DTYPE,
            shape=(self.chunk_count,),
        )
        self.videos = np.memmap(
            self.dataset_base_path / videos_name,
            mode="r",
            dtype=VIDEO_DTYPE,
            shape=(self.chunk_count,),
        )
        self.indices = np.asarray(
            list(indices) if indices is not None else np.arange(self.chunk_count),
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        frames = np.asarray(self.samples[idx], dtype=np.float32)
        # frames = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
        frames = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).contiguous()
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        video_id = torch.tensor(int(self.videos[idx]), dtype=torch.long)
        return frames, label, video_id


class C3DBackbone(nn.Module):
    _TAIL_LAYERS = ("conv4a", "conv4b", "conv5a", "conv5b", "fc6", "fc7", "fc8")

    def __init__(self, output_layer: str = "fc6"):
        super().__init__()
        if output_layer not in {"pool3", "fc6", "fc7", "fc8"}:
            raise ValueError("output_layer must be one of: pool3, fc6, fc7, fc8")

        self.output_layer = output_layer
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        if output_layer == "pool3":
            return

        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc6 = nn.Linear(8192, 4096)
        if output_layer == "fc6":
            return

        self.dropout = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096)
        if output_layer == "fc8":
            self.fc8 = nn.Linear(4096, 487)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # PyTorch passes a private copy of the state mapping here. Legacy
        # checkpoints contain these weights even when the forward skips them.
        for layer_name in self._TAIL_LAYERS:
            if layer_name not in self._modules:
                state_dict.pop(f"{prefix}{layer_name}.weight", None)
                state_dict.pop(f"{prefix}{layer_name}.bias", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        if self.output_layer == "pool3":
            return x

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = nn.functional.pad(x, (0, 1, 0, 1, 0, 0))
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc6(x))
        if self.output_layer == "fc6":
            return x

        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        if self.output_layer == "fc7":
            return x

        x = self.dropout(x)
        return self.fc8(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None):
        if state is None:
            size = (x.size(0), self.hidden_channels, x.size(2), x.size(3))
            h = x.new_zeros(size)
            c = x.new_zeros(size)
        else:
            h, c = state

        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class C3DCNNClassifier(nn.Module):
    def __init__(self, n_classes: int = 3, freeze_backbone: bool = True):
        super().__init__()
        self.c3d = C3DBackbone(output_layer="fc6")
        if freeze_backbone:
            for parameter in self.c3d.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.c3d(x))


class C3DCNNLSTMClassifier(nn.Module):
    def __init__(self, n_classes: int = 3, freeze_backbone: bool = True):
        super().__init__()
        self.c3d = C3DBackbone(output_layer="pool3")
        if freeze_backbone:
            for parameter in self.c3d.parameters():
                parameter.requires_grad = False

        self.conv_lstm = ConvLSTMCell(256, 64, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c3d(x)
        state = None
        for t in range(x.size(2)):
            state = self.conv_lstm(x[:, :, t], state)

        h, _ = state
        x = self.batch_norm(h)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class FrameConvLSTMClassifier(nn.Module):
    def __init__(self, n_classes: int = 3, freeze_backbone: bool | None = None):
        super().__init__()
        self.conv_lstm = ConvLSTMCell(3, 64, kernel_size=3, padding=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(64 * 112 * 112, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = None
        sequence = x.permute(0, 2, 1, 3, 4)
        for t in range(sequence.size(1)):
            state = self.conv_lstm(sequence[:, t], state)

        h, _ = state
        return self.classifier(h)


def _find_keras_dataset(h5_file, layer_name: str, weight_name: str):
    import h5py

    matches = []

    def visitor(name, obj):
        parts = name.split("/")
        if isinstance(obj, h5py.Dataset) and layer_name in parts and parts[-1].startswith(weight_name):
            matches.append(obj)

    h5_file.visititems(visitor)
    if not matches:
        return None
    return matches[0][()]


def load_keras_c3d_weights(
    backbone: C3DBackbone,
    weights_path: Path | str,
    include_fc: bool = True,
    strict_shapes: bool = False,
) -> list[str]:
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install h5py to convert the Keras C3D .h5 weights: pip install h5py") from exc

    import h5py

    weights_path = Path(weights_path)
    loaded_layers = []
    conv_layers = ["conv1", "conv2", "conv3a", "conv3b", "conv4a", "conv4b", "conv5a", "conv5b"]
    dense_layers = ["fc6", "fc7", "fc8"] if include_fc else []

    with h5py.File(weights_path, "r") as h5_file:
        for layer_name in conv_layers:
            layer = getattr(backbone, layer_name, None)
            if layer is None:
                continue
            kernel = _find_keras_dataset(h5_file, layer_name, "kernel")
            bias = _find_keras_dataset(h5_file, layer_name, "bias")
            if kernel is None or bias is None:
                continue

            kernel = torch.from_numpy(np.transpose(kernel, (4, 3, 0, 1, 2))).float()
            bias = torch.from_numpy(bias).float()
            if kernel.shape == layer.weight.shape and bias.shape == layer.bias.shape:
                layer.weight.data.copy_(kernel)
                layer.bias.data.copy_(bias)
                loaded_layers.append(layer_name)
            elif strict_shapes:
                raise ValueError(f"Shape mismatch for {layer_name}: {kernel.shape} vs {layer.weight.shape}")

        for layer_name in dense_layers:
            layer = getattr(backbone, layer_name, None)
            if layer is None:
                continue
            kernel = _find_keras_dataset(h5_file, layer_name, "kernel")
            bias = _find_keras_dataset(h5_file, layer_name, "bias")
            if kernel is None or bias is None:
                continue

            kernel = torch.from_numpy(np.transpose(kernel, (1, 0))).float()
            bias = torch.from_numpy(bias).float()
            if kernel.shape == layer.weight.shape and bias.shape == layer.bias.shape:
                layer.weight.data.copy_(kernel)
                layer.bias.data.copy_(bias)
                loaded_layers.append(layer_name)
            elif strict_shapes:
                raise ValueError(f"Shape mismatch for {layer_name}: {kernel.shape} vs {layer.weight.shape}")

    return loaded_layers


def build_model(
    model_name: str,
    freeze_backbone: bool = True,
    n_classes: int = 3,
    keras_c3d_weights_path: Path | str | None = None,
    verbose: bool = True,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name in {"3dcnnlstm", "c3d_lstm", "c3dcnnlstm"}:
        model = C3DCNNLSTMClassifier(n_classes=n_classes, freeze_backbone=freeze_backbone)
        include_fc = False
    elif model_name in {"c3dcnn", "c3dfc", "c3d"}:
        model = C3DCNNClassifier(n_classes=n_classes, freeze_backbone=freeze_backbone)
        include_fc = True
    elif model_name in {"lstm", "convlstm"}:
        model = FrameConvLSTMClassifier(n_classes=n_classes, freeze_backbone=freeze_backbone)
        include_fc = False
    elif model_name == "mvit_v2_s":
        from video_models import getMViTv2SModel

        # Inference/checkpoint construction must not download pretrained weights.
        model = getMViTv2SModel(weights=None, n_classes=n_classes,
                                freeze_backbone=freeze_backbone, verbose=False)
        include_fc = False
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if keras_c3d_weights_path is not None and hasattr(model, "c3d"):
        loaded = load_keras_c3d_weights(model.c3d, keras_c3d_weights_path, include_fc=include_fc)
        if verbose:
            print(f"Loaded Keras C3D layers into PyTorch: {loaded}")

    if verbose:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        print(f"{model.__class__.__name__}: {trainable:,} trainable / {total:,} total parameters")

    return model


def get3DCNNLSTMModel(verbose: bool = True, freeze_backbone: bool = True, lr: float = 1e-4):
    root = get_project_root()
    return build_model(
        "3dcnnlstm",
        freeze_backbone=freeze_backbone,
        keras_c3d_weights_path=root / "c3d_weight" / "3dcnn_weights.h5",
        verbose=verbose,
    )


def getC3DCNNModel(verbose: bool = True, freeze_backbone: bool = True, lr: float = 1e-4):
    root = get_project_root()
    return build_model(
        "c3dcnn",
        freeze_backbone=freeze_backbone,
        keras_c3d_weights_path=root / "c3d_weight" / "3dcnn_weights.h5",
        verbose=verbose,
    )


def getLSTMModel(verbose: bool = True, freeze_backbone: bool | None = None, lr: float = 1e-4):
    return build_model("lstm", freeze_backbone=False, verbose=verbose)


def aggregate_video_prediction_two_step(
    chunk_probas: np.ndarray,
    violence_presence_threshold: float = DEFAULT_VIOLENCE_PRESENCE_THRESHOLD,
    violent_chunk_threshold: float = DEFAULT_VIOLENT_CHUNK_THRESHOLD,
    min_violent_chunk_fraction: float = DEFAULT_MIN_VIOLENT_CHUNK_FRACTION,
    violence_mean_margin: float = DEFAULT_VIOLENCE_MEAN_MARGIN,
    high_violence_evidence_threshold: float = DEFAULT_HIGH_VIOLENCE_EVIDENCE_THRESHOLD,
    high_violence_mean_threshold: float = DEFAULT_HIGH_VIOLENCE_MEAN_THRESHOLD,
    high_violence_margin: float = DEFAULT_HIGH_VIOLENCE_MARGIN,
    mean_weight: float = DEFAULT_VIDEO_MEAN_WEIGHT,
    max_weight: float = DEFAULT_VIDEO_MAX_WEIGHT,
) -> dict:
    chunk_probas = np.asarray(chunk_probas, dtype=np.float64)
    if chunk_probas.ndim != 2 or chunk_probas.shape[1] != 3:
        raise ValueError(f"Expected chunk probabilities with shape (chunks, 3), got {chunk_probas.shape}")
    if chunk_probas.shape[0] == 0:
        raise ValueError("Cannot aggregate an empty set of chunk probabilities")
    for name, value in {
        "violence_presence_threshold": violence_presence_threshold,
        "violent_chunk_threshold": violent_chunk_threshold,
        "min_violent_chunk_fraction": min_violent_chunk_fraction,
        "high_violence_evidence_threshold": high_violence_evidence_threshold,
        "high_violence_mean_threshold": high_violence_mean_threshold,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if violence_mean_margin < 0:
        raise ValueError("violence_mean_margin must be non-negative")
    if high_violence_margin < 0:
        raise ValueError("high_violence_margin must be non-negative")

    weight_sum = mean_weight + max_weight
    if weight_sum <= 0:
        raise ValueError("mean_weight + max_weight must be positive")
    mean_weight = mean_weight / weight_sum
    max_weight = max_weight / weight_sum

    mean_prob = np.mean(chunk_probas, axis=0)
    max_prob = np.max(chunk_probas, axis=0)
    strongest_violence_chunk_prob = np.max(chunk_probas[:, 1:3], axis=1)
    violence_mean = float(np.mean(strongest_violence_chunk_prob))
    violence_max = float(np.max(strongest_violence_chunk_prob))
    strong_violence_fraction = float(np.mean(strongest_violence_chunk_prob >= violent_chunk_threshold))
    nonviolence_mean = float(mean_prob[0])

    violence_present = (
        violence_max >= violence_presence_threshold
        or (
            violence_mean >= nonviolence_mean + violence_mean_margin
            and strong_violence_fraction >= min_violent_chunk_fraction
        )
    )

    class_evidence = (mean_weight * mean_prob) + (max_weight * max_prob)
    video_evidence = class_evidence.copy()
    video_evidence[0] = nonviolence_mean

    high_margin = float(video_evidence[2] - video_evidence[1])
    high_confident = (
        video_evidence[2] >= high_violence_evidence_threshold
        or (
            mean_prob[2] >= high_violence_mean_threshold
            and high_margin >= high_violence_margin
        )
    )

    if not violence_present:
        pred_id = 0
    elif high_confident:
        pred_id = 2
    else:
        pred_id = 1

    return {
        "pred_id": pred_id,
        "video_evidence": video_evidence,
        "mean_prob": mean_prob,
        "max_prob": max_prob,
        "violence_present": bool(violence_present),
        "violence_mean": violence_mean,
        "violence_max": violence_max,
        "strong_violence_fraction": strong_violence_fraction,
        "nonviolence_mean": nonviolence_mean,
        "high_margin": high_margin,
        "high_confident": bool(high_confident),
        "violence_presence_threshold": violence_presence_threshold,
        "violent_chunk_threshold": violent_chunk_threshold,
        "min_violent_chunk_fraction": min_violent_chunk_fraction,
        "violence_mean_margin": violence_mean_margin,
        "high_violence_evidence_threshold": high_violence_evidence_threshold,
        "high_violence_mean_threshold": high_violence_mean_threshold,
        "high_violence_margin": high_violence_margin,
        "mean_weight": mean_weight,
        "max_weight": max_weight,
    }


def video_ensemble_two_step(
    y_true_chunks: np.ndarray,
    probas_chunks: np.ndarray,
    video_ids: np.ndarray,
    n_classes: int,
    violence_presence_threshold: float = DEFAULT_VIOLENCE_PRESENCE_THRESHOLD,
    violent_chunk_threshold: float = DEFAULT_VIOLENT_CHUNK_THRESHOLD,
    min_violent_chunk_fraction: float = DEFAULT_MIN_VIOLENT_CHUNK_FRACTION,
    violence_mean_margin: float = DEFAULT_VIOLENCE_MEAN_MARGIN,
    high_violence_evidence_threshold: float = DEFAULT_HIGH_VIOLENCE_EVIDENCE_THRESHOLD,
    high_violence_mean_threshold: float = DEFAULT_HIGH_VIOLENCE_MEAN_THRESHOLD,
    high_violence_margin: float = DEFAULT_HIGH_VIOLENCE_MARGIN,
    mean_weight: float = DEFAULT_VIDEO_MEAN_WEIGHT,
    max_weight: float = DEFAULT_VIDEO_MAX_WEIGHT,
    return_details: bool = False,
):
    if n_classes != 3:
        raise ValueError("The two-step video ensemble expects 3 classes: non, low, high")
    y_true_chunks = np.asarray(y_true_chunks).astype(int)
    probas_chunks = np.asarray(probas_chunks)
    video_ids = np.asarray(video_ids)

    video_true_list = []
    video_pred_list = []
    detail_list = []
    for video_id in np.unique(video_ids):
        idx = video_ids == video_id
        video_true = np.argmax(np.bincount(y_true_chunks[idx], minlength=n_classes))
        aggregation = aggregate_video_prediction_two_step(
            probas_chunks[idx],
            violence_presence_threshold=violence_presence_threshold,
            violent_chunk_threshold=violent_chunk_threshold,
            min_violent_chunk_fraction=min_violent_chunk_fraction,
            violence_mean_margin=violence_mean_margin,
            high_violence_evidence_threshold=high_violence_evidence_threshold,
            high_violence_mean_threshold=high_violence_mean_threshold,
            high_violence_margin=high_violence_margin,
            mean_weight=mean_weight,
            max_weight=max_weight,
        )
        video_pred = aggregation["pred_id"]
        video_true_list.append(video_true)
        video_pred_list.append(video_pred)
        detail_list.append({"video_id": video_id, **aggregation})

    video_true_arr = np.array(video_true_list)
    video_pred_arr = np.array(video_pred_list)
    accuracy = np.mean(video_true_arr == video_pred_arr)
    if return_details:
        return accuracy, video_true_arr, video_pred_arr, detail_list
    return accuracy, video_true_arr, video_pred_arr


def video_ensemble_mean_prob(
    y_true_chunks: np.ndarray,
    probas_chunks: np.ndarray,
    video_ids: np.ndarray,
    n_classes: int,
):
    return video_ensemble_two_step(y_true_chunks, probas_chunks, video_ids, n_classes)


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels, _ in data_loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def predict_proba(model, data_loader, device):
    model.eval()
    probas = []
    labels = []
    videos = []
    running_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for frames, batch_labels, batch_videos in data_loader:
        frames = frames.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        logits = model(frames)
        loss = criterion(logits, batch_labels)
        probabilities = torch.softmax(logits, dim=1)

        running_loss += loss.item() * batch_labels.size(0)
        total += batch_labels.size(0)
        probas.append(probabilities.cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
        videos.append(batch_videos.numpy())

    return (
        running_loss / total,
        np.concatenate(probas, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(videos, axis=0),
    )


def evaluate_loader(model, data_loader, device):
    loss, probas, labels, videos = predict_proba(model, data_loader, device)
    predictions = np.argmax(probas, axis=1)
    return loss, np.mean(labels == predictions), probas, labels, videos


def make_loader(dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _training_components(get_model, device, verbose, lr, model_kwargs=None, optimizer_factory=None):
    """Optional per-model settings; absent overrides preserve the original path."""
    kwargs = {"verbose": verbose, "freeze_backbone": True, "lr": lr}
    kwargs.update(model_kwargs or {})
    model = get_model(**kwargs).to(device)
    parameters = (p for p in model.parameters() if p.requires_grad)
    optimizer = (optimizer_factory or torch.optim.Adam)(parameters, lr=lr)
    return model, optimizer


def run_end_to_end_experiment(
    get_model: Callable[..., nn.Module],
    batch_size: int,
    dataset_base_path: Path | str,
    mmap_dataset_base_path: Path | str,
    samples_mmap_name: str,
    labels_mmap_name: str,
    end_to_end_model_name: str,
    r_state: int,
    class_names: list[str] | None = None,
    epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-5,
    n_splits: int = 10,
    num_workers: int = 0,
    *,
    model_kwargs: dict | None = None,
    optimizer_factory: Callable | None = None,
):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    from sklearn.preprocessing import label_binarize

    class_names = class_names or CLASS_NAMES
    device = get_device()
    chunk_number = infer_mmap_chunk_count(
        mmap_dataset_base_path,
        labels_name=labels_mmap_name,
        samples_name=samples_mmap_name,
    )

    full_dataset = MMapVideoChunkDataset(
        mmap_dataset_base_path,
        samples_name=samples_mmap_name,
        labels_name=labels_mmap_name,
        chunk_count=chunk_number,
    )
    y = np.asarray(full_dataset.labels)
    video_ids = np.asarray(full_dataset.videos)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=r_state)
    all_chunk_val_curves = []
    all_video_val_curves = []
    scores = []
    train_accs = np.zeros(n_splits)
    macro_precisions = np.zeros(n_splits)
    macro_recalls = np.zeros(n_splits)
    macro_f1s = np.zeros(n_splits)
    macro_aucs = np.zeros(n_splits)
    video_accs = np.zeros(n_splits)

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(chunk_number), y, groups=video_ids), start=1):
        train_labels = y[train_idx]
        train_videos = video_ids[train_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=r_state)
        tr_local, val_local = next(gss.split(np.zeros(len(train_idx)), train_labels, groups=train_videos))
        tr_idx = train_idx[tr_local]
        val_idx = train_idx[val_local]

        print(f"\n[Fold {fold}] Dataset sizes and distribution:")
        print(f"Train: {len(tr_idx)}, dist: {Counter(y[tr_idx])}")
        print(f"Val:   {len(val_idx)}, dist: {Counter(y[val_idx])}")
        print(f"Test:  {len(test_idx)}, dist: {Counter(y[test_idx])}")

        train_loader = make_loader(
            MMapVideoChunkDataset(mmap_dataset_base_path, samples_mmap_name, labels_mmap_name, indices=tr_idx, chunk_count=chunk_number),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = make_loader(
            MMapVideoChunkDataset(mmap_dataset_base_path, samples_mmap_name, labels_mmap_name, indices=val_idx, chunk_count=chunk_number),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = make_loader(
            MMapVideoChunkDataset(mmap_dataset_base_path, samples_mmap_name, labels_mmap_name, indices=test_idx, chunk_count=chunk_number),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model, optimizer = _training_components(
            get_model, device, fold == 1, lr, model_kwargs, optimizer_factory
        )
        criterion = nn.CrossEntropyLoss()
        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        chunk_curve = []
        video_curve = []
        best_train_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_probas, val_labels, val_videos = evaluate_loader(model, val_loader, device)
            val_video_acc, _, _ = video_ensemble_two_step(val_labels, val_probas, val_videos, len(class_names))
            chunk_curve.append(val_acc)
            video_curve.append(val_video_acc)

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | val_chunk_acc={val_acc:.4f} | val_video_acc={val_video_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        all_chunk_val_curves.append(chunk_curve)
        all_video_val_curves.append(video_curve)
        train_accs[fold - 1] = best_train_acc

        test_loss, test_acc, test_probas, test_labels, test_videos = evaluate_loader(model, test_loader, device)
        y_pred = np.argmax(test_probas, axis=1)
        scores.append([test_loss, test_acc])

        video_acc, video_true_arr, video_pred_arr = video_ensemble_two_step(
            test_labels, test_probas, test_videos, len(class_names)
        )
        video_accs[fold - 1] = video_acc
        print(f"[Video TwoStep] split {fold} | video-acc={video_acc:.4f}")
        print("Video-level Confusion Matrix")
        print(confusion_matrix(video_true_arr, video_pred_arr))
        print("Video-level Classification Report")
        print(classification_report(video_true_arr, video_pred_arr, target_names=class_names))

        report = classification_report(test_labels, y_pred, target_names=class_names, output_dict=True)
        macro_precisions[fold - 1] = report["macro avg"]["precision"]
        macro_recalls[fold - 1] = report["macro avg"]["recall"]
        macro_f1s[fold - 1] = report["macro avg"]["f1-score"]
        try:
            y_true_bin = label_binarize(test_labels, classes=list(range(len(class_names))))
            macro_aucs[fold - 1] = roc_auc_score(y_true_bin, test_probas, average="macro", multi_class="ovr")
        except Exception:
            macro_aucs[fold - 1] = np.nan

        print(f"Confusion matrix split {fold}")
        print(confusion_matrix(test_labels, y_pred))
        print(classification_report(test_labels, y_pred, target_names=class_names))
        print(f"Loss: {test_loss}")
        print(f"Accuracy: {test_acc}")
        print(f"Macro Precision: {macro_precisions[fold - 1]}")
        print(f"Macro Recall: {macro_recalls[fold - 1]}")
        print(f"Macro F1: {macro_f1s[fold - 1]}")
        print(f"Macro ROC-AUC (OvR): {macro_aucs[fold - 1]}")

    scores = np.array(scores, dtype=float)
    losses = scores[:, 0]
    accuracies = scores[:, 1]
    print("Video accuracy per split")
    print(video_accs)
    print(f"Avg video accuracy: {np.mean(video_accs)} +/- {np.std(video_accs)}")
    print("Train accuracy per split")
    print(train_accs)
    print("Losses")
    print(losses)
    print("Accuracies")
    print(accuracies)
    print("Macro Precisions")
    print(macro_precisions)
    print("Macro Recalls")
    print(macro_recalls)
    print("Macro F1-scores")
    print(macro_f1s)
    print("Macro ROC-AUC (OvR)")
    print(macro_aucs)
    print(f"Avg train accuracy: {np.mean(train_accs)} +/- {np.std(train_accs)}")
    print(f"Avg loss: {np.mean(losses)} +/- {np.std(losses)}")
    print(f"Avg accuracy: {np.mean(accuracies)} +/- {np.std(accuracies)}")
    print(f"Avg macro precision: {np.mean(macro_precisions)} +/- {np.std(macro_precisions)}")
    print(f"Avg macro recall: {np.mean(macro_recalls)} +/- {np.std(macro_recalls)}")
    print(f"Avg macro f1: {np.mean(macro_f1s)} +/- {np.std(macro_f1s)}")
    print(f"Avg macro auc: {np.nanmean(macro_aucs)} +/- {np.nanstd(macro_aucs)}")

    return all_chunk_val_curves, all_video_val_curves


def plot_validation_curves(all_chunk_val_curves, all_video_val_curves):
    max_len = max(len(curve) for curve in all_chunk_val_curves)
    chunk_mat = np.full((len(all_chunk_val_curves), max_len), np.nan)
    video_mat = np.full((len(all_video_val_curves), max_len), np.nan)

    for index, curve in enumerate(all_chunk_val_curves):
        chunk_mat[index, : len(curve)] = curve
    for index, curve in enumerate(all_video_val_curves):
        video_mat[index, : len(curve)] = curve

    epochs_axis = np.arange(1, max_len + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, np.nanmean(chunk_mat, axis=0), label="Avg validation chunk accuracy")
    plt.plot(epochs_axis, np.nanmean(video_mat, axis=0), label="Avg validation video accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Average Validation Accuracy Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def train_final_model(
    get_model: Callable[..., nn.Module],
    dataset_base_path: Path | str,
    checkpoint_path: Path | str,
    batch_size: int = 4,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-5,
    validation_size: float = 0.1,
    random_state: int = 42,
    num_workers: int = 0,
    *,
    model_kwargs: dict | None = None,
    optimizer_factory: Callable | None = None,
):
    from sklearn.model_selection import GroupShuffleSplit

    device = get_device()
    dataset_base_path = Path(dataset_base_path)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_number = infer_mmap_chunk_count(dataset_base_path)

    full_dataset = MMapVideoChunkDataset(dataset_base_path, chunk_count=chunk_number)
    y = np.asarray(full_dataset.labels)
    videos = np.asarray(full_dataset.videos)

    gss = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(np.zeros(chunk_number), y, groups=videos))
    print("Train:", len(train_idx), Counter(y[train_idx]))
    print("Val:  ", len(val_idx), Counter(y[val_idx]))
    print("Train videos:", len(np.unique(videos[train_idx])))
    print("Val videos:  ", len(np.unique(videos[val_idx])))

    train_loader = make_loader(
        MMapVideoChunkDataset(dataset_base_path, indices=train_idx, chunk_count=chunk_number),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        MMapVideoChunkDataset(dataset_base_path, indices=val_idx, chunk_count=chunk_number),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model, optimizer = _training_components(
        get_model, device, True, lr, model_kwargs, optimizer_factory
    )
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "val_video_accuracy": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_probas, val_labels, val_videos = evaluate_loader(model, val_loader, device)
        val_video_acc, _, _ = video_ensemble_two_step(val_labels, val_probas, val_videos, len(CLASS_NAMES))

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_video_accuracy"].append(val_video_acc)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_chunk_acc={val_acc:.4f} | val_video_acc={val_video_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_class": model.__class__.__name__,
                "class_names": CLASS_NAMES,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_video_accuracy": val_video_acc,
            }
            metadata = getattr(model, "checkpoint_metadata", None)
            if callable(metadata):
                checkpoint["model_metadata"] = metadata()
                checkpoint["training_config"] = {
                    "lr": lr, "batch_size": batch_size, "epochs": epochs,
                    "patience": patience, "validation_size": validation_size,
                    "random_state": random_state, "dataset_base_path": str(dataset_base_path),
                    "optimizer": optimizer.__class__.__name__,
                    "optimizer_defaults": copy.deepcopy(optimizer.defaults),
                    "model_kwargs": copy.deepcopy(model_kwargs or {}),
                    "split": "GroupShuffleSplit by existing video ID",
                    "video_aggregation": "two_step",
                    "video_aggregation_defaults": {
                        "violence_presence_threshold": DEFAULT_VIOLENCE_PRESENCE_THRESHOLD,
                        "violent_chunk_threshold": DEFAULT_VIOLENT_CHUNK_THRESHOLD,
                        "min_violent_chunk_fraction": DEFAULT_MIN_VIOLENT_CHUNK_FRACTION,
                        "violence_mean_margin": DEFAULT_VIOLENCE_MEAN_MARGIN,
                        "high_violence_evidence_threshold": DEFAULT_HIGH_VIOLENCE_EVIDENCE_THRESHOLD,
                        "high_violence_mean_threshold": DEFAULT_HIGH_VIOLENCE_MEAN_THRESHOLD,
                        "high_violence_margin": DEFAULT_HIGH_VIOLENCE_MARGIN,
                        "mean_weight": DEFAULT_VIDEO_MEAN_WEIGHT,
                        "max_weight": DEFAULT_VIDEO_MAX_WEIGHT,
                    },
                }
            torch.save(checkpoint, checkpoint_path)
            epochs_without_improvement = 0
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    return model, history


def plot_final_training_history(history):
    plt.figure()
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.show()

    plt.figure()
    plt.plot(history["accuracy"], label="train_acc")
    plt.plot(history["val_accuracy"], label="val_acc")
    plt.plot(history["val_video_accuracy"], label="val_video_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.show()


def load_checkpoint(model: nn.Module, checkpoint_path: Path | str, device: torch.device | None = None):
    device = device or get_device()
    # Stage weights on CPU so legacy checkpoint tensors never occupy GPU memory.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def count_strided_chunks(frame_count: int) -> int:
    if frame_count <= 0:
        return 0
    if frame_count < RAW_FRAMES_PER_CHUNK:
        return 1
    return ((frame_count - RAW_FRAMES_PER_CHUNK) // CHUNK_FRAME_STEP) + 1


def iter_chunk_start_frames(frame_count: int):
    if frame_count <= 0:
        return
    if frame_count < RAW_FRAMES_PER_CHUNK:
        yield 0
        return
    max_start = frame_count - RAW_FRAMES_PER_CHUNK
    for start_frame in range(0, max_start + 1, CHUNK_FRAME_STEP):
        yield start_frame


def sampled_frame_indices(start_frame: int, frame_count: int) -> list[int]:
    last_frame = max(0, frame_count - 1)
    return [min(start_frame + offset * FRAME_SAMPLE_STRIDE, last_frame) for offset in range(FRAMES_PER_CHUNK)]


def frames_to_chunks(video_path: Path | str):
    video_path = Path(video_path)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)))
    video.release()

    if len(frames) == 0:
        return None, frame_count, fps

    chunk_arrays = []
    for start_frame in iter_chunk_start_frames(len(frames)):
        frame_indices = sampled_frame_indices(start_frame, len(frames))
        chunk_arrays.append(np.asarray([frames[index] for index in frame_indices], dtype=np.float32))

    if len(chunk_arrays) == 0:
        return None, frame_count, fps

    x = np.stack(chunk_arrays, axis=0)
    x = torch.from_numpy(x).permute(0, 4, 1, 2, 3).contiguous()
    return x, frame_count, fps


@torch.no_grad()
def run_real_world_video_inference(
    video_path: Path | str,
    checkpoint_path: Path | str,
    model_name: str = "c3dcnn",
    class_names: list[str] | None = None,
    freeze_backbone: bool = True,
    batch_size: int = 8,
    verbose: bool = True,
    violence_presence_threshold: float = DEFAULT_VIOLENCE_PRESENCE_THRESHOLD,
    violent_chunk_threshold: float = DEFAULT_VIOLENT_CHUNK_THRESHOLD,
    min_violent_chunk_fraction: float = DEFAULT_MIN_VIOLENT_CHUNK_FRACTION,
    violence_mean_margin: float = DEFAULT_VIOLENCE_MEAN_MARGIN,
    high_violence_evidence_threshold: float = DEFAULT_HIGH_VIOLENCE_EVIDENCE_THRESHOLD,
    high_violence_mean_threshold: float = DEFAULT_HIGH_VIOLENCE_MEAN_THRESHOLD,
    high_violence_margin: float = DEFAULT_HIGH_VIOLENCE_MARGIN,
    video_mean_weight: float = DEFAULT_VIDEO_MEAN_WEIGHT,
    video_max_weight: float = DEFAULT_VIDEO_MAX_WEIGHT,
):
    class_names = class_names or CLASS_NAMES
    device = get_device()
    model = build_model(model_name, freeze_backbone=freeze_backbone, n_classes=len(class_names), verbose=verbose)
    model, _ = load_checkpoint(model, checkpoint_path, device)

    video_path = Path(video_path)
    if video_path.is_dir():
        video_files = []
        for extension in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_files.extend(video_path.glob(extension))
        video_files = sorted(video_files)
    else:
        video_files = [video_path]

    if len(video_files) == 0:
        print("No video files found.")
        return []

    all_results = []
    for video_file in video_files:
        print("\n==============================")
        print(f"Processing: {video_file.name}")
        chunks, frame_count, fps = frames_to_chunks(video_file)
        if chunks is None:
            print("Too short, skip.")
            continue

        if verbose:
            print(f"*** [Video Info] frames: {frame_count} - fps: {fps} - chunks: {chunks.size(0)}")
            print(f"Input shape: {tuple(chunks.shape)}")

        model.eval()
        probabilities = []
        for start in range(0, chunks.size(0), batch_size):
            batch = chunks[start : start + batch_size].to(device)
            logits = model(batch)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

        chunk_probas = np.concatenate(probabilities, axis=0)
        video_aggregation = aggregate_video_prediction_two_step(
            chunk_probas,
            violence_presence_threshold=violence_presence_threshold,
            violent_chunk_threshold=violent_chunk_threshold,
            min_violent_chunk_fraction=min_violent_chunk_fraction,
            violence_mean_margin=violence_mean_margin,
            high_violence_evidence_threshold=high_violence_evidence_threshold,
            high_violence_mean_threshold=high_violence_mean_threshold,
            high_violence_margin=high_violence_margin,
            mean_weight=video_mean_weight,
            max_weight=video_max_weight,
        )
        video_prob = video_aggregation["mean_prob"]
        video_evidence = video_aggregation["video_evidence"]
        video_pred_id = int(video_aggregation["pred_id"])
        video_pred_label = class_names[video_pred_id]

        print(f"Video: {video_file.name}")
        print(f"Prediction: {video_pred_label}")
        print(f"Mean prob: {video_prob}")
        print(f"Max prob: {video_aggregation['max_prob']}")
        print(f"Two-step evidence: {video_evidence}")
        print(
            "Violence gate: "
            f"present={video_aggregation['violence_present']} | "
            f"mean={video_aggregation['violence_mean']:.4f} | "
            f"max={video_aggregation['violence_max']:.4f} | "
            f"strong_fraction={video_aggregation['strong_violence_fraction']:.4f} | "
            f"high_confident={video_aggregation['high_confident']} | "
            f"high_margin={video_aggregation['high_margin']:.4f}"
        )

        all_results.append(
            {
                "video_name": video_file.name,
                "video_path": str(video_file),
                "video_prob": video_prob,
                "video_evidence": video_evidence,
                "video_max_prob": video_aggregation["max_prob"],
                "violence_present": video_aggregation["violence_present"],
                "violence_mean": video_aggregation["violence_mean"],
                "violence_max": video_aggregation["violence_max"],
                "strong_violence_fraction": video_aggregation["strong_violence_fraction"],
                "high_confident": video_aggregation["high_confident"],
                "high_margin": video_aggregation["high_margin"],
                "video_pred_id": video_pred_id,
                "video_pred_label": video_pred_label,
                "chunk_probabilities": chunk_probas,
            }
        )

    return all_results
