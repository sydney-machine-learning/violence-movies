"""Inference benchmarks with importable model/input/state adapters.

The default worker is a fresh Python process per case. It never trains a model,
edits a checkpoint, prunes a model, or changes the training implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import torch
from torch import nn


@dataclass
class Factory:
    """An importable ``module:object`` plus JSON-serializable keyword arguments."""

    target: str
    kwargs: dict = field(default_factory=dict)

    def create(self, **runtime_kwargs):
        module_name, separator, object_name = self.target.partition(":")
        if not separator:
            raise ValueError(f"Expected module:object, got {self.target!r}")
        obj = importlib.import_module(module_name)
        for part in object_name.split("."):
            obj = getattr(obj, part)
        return obj(**{**self.kwargs, **runtime_kwargs})


@dataclass
class BenchmarkCase:
    name: str
    model_factory: Factory
    checkpoint: str | None = None
    factory_weights_description: str | None = None
    # Default input creation is synthetic, prepared outside the measured forward.
    input_shape: tuple[int, ...] = (1, 3, 16, 112, 112)
    input_range: tuple[float, float] = (0.0, 255.0)
    input_factory: Factory | None = None
    adapter_factory: Factory = field(default_factory=lambda: Factory("model_benchmark:ForwardAdapter"))
    checkpoint_loader: Factory | None = None
    parameter_selector: Factory | None = None
    mac_counter: Factory | None = None
    mode: str = "clip"
    state_policy: str = "reset_each_call"
    batch_size: int = 1
    frames_per_call: int = 16  # New input frames, per batch item, not cached frames.
    input_description: str = "BCTHW; BGR; original pixel scale"
    comparison_group: str = "16_frames_112x112"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.checkpoint and self.factory_weights_description:
            raise ValueError("Use a checkpoint or explicitly declared factory-provided weights, not both")
        if self.mode not in {"clip", "stream"}:
            raise ValueError("mode must be 'clip' or 'stream'")
        if self.state_policy not in {"reset_each_call", "carry"}:
            raise ValueError("state_policy must be 'reset_each_call' or 'carry'")
        if self.mode == "clip" and self.state_policy != "reset_each_call":
            raise ValueError("Independent clips must reset state each call")
        if self.mode == "stream" and self.adapter_factory.target == "model_benchmark:ForwardAdapter":
            raise ValueError("Streaming cases require an explicit adapter that resets their cache/state")
        if self.batch_size < 1 or self.frames_per_call < 1:
            raise ValueError("batch_size and frames_per_call must be positive")
        if not self.input_factory and (
            not self.input_shape or self.input_shape[0] != self.batch_size
        ):
            raise ValueError("The default input's first dimension must equal batch_size")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ["model_factory", "input_factory", "adapter_factory", "checkpoint_loader",
                    "parameter_selector", "mac_counter"]:
            if data.get(key) is not None:
                data[key] = Factory(**data[key])
        return cls(**data)


@dataclass
class BenchmarkConfig:
    device: str = "auto"
    dtype: str = "float32"
    warmup: int = 30
    iterations: int = 200
    trials: int = 3
    seed: int = 42
    cpu_threads: int = 1
    allow_untrained: bool = False
    allow_tf32: bool = False
    hash_checkpoints: bool = True
    timeout_seconds: int = 900

    def __post_init__(self):
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported floating-point dtype")
        if self.warmup < 0 or min(self.iterations, self.trials, self.cpu_threads, self.timeout_seconds) < 1:
            raise ValueError("Require nonnegative warmup and positive iteration/trial/thread/timeout counts")


class ForwardAdapter:
    """Default stateless adapter; accepts tensors, positional tuples or keyword dicts.

    A streaming adapter should override reset() and step(), keeping any external
    cache on this adapter or resetting the model's internal cache in reset().
    """

    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def step(self, inputs):
        if isinstance(inputs, dict):
            return self.model(**inputs)
        if isinstance(inputs, tuple):
            return self.model(*inputs)
        return self.model(inputs)


def cnn_active_parameter_names(model):
    """Explicit selector for this project's three models, including frozen weights."""
    from cnn_lstm_pytorch import C3DCNNClassifier, C3DCNNLSTMClassifier, FrameConvLSTMClassifier

    if not isinstance(model, (C3DCNNClassifier, C3DCNNLSTMClassifier, FrameConvLSTMClassifier)):
        raise TypeError("Supply a parameter selector for this architecture")
    return [name for name, _ in model.named_parameters()]


def count_conv_linear_macs(model, adapter, inputs):
    """Count only module Conv1d/2d/3d and Linear MACs, on every invocation.

    This deliberately does NOT claim total FLOPs. Functional/attention operations,
    normalization, pooling, nonlinearities and elementwise gates are excluded.
    """
    count = 0
    calls = {}
    handles = []

    def hook(module, args, output):
        nonlocal count
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            count += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            count += output.numel() * module.in_features
        else:
            return
        key = type(module).__name__
        calls[key] = calls.get(key, 0) + 1

    try:
        for module in model.modules():
            handles.append(module.register_forward_hook(hook))
        adapter.step(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return {"macs_per_call": count, "scope": "module convolutions and Linear only; not total FLOPs", "calls": calls}


MAC_SCOPE = (
    "Dense MACs for convolution, linear and matrix contractions, including attention and relative position terms. "
    "Excludes bias, elementwise arithmetic, activations, normalization, softmax, pooling and interpolation."
)

# Forward operators outside the declared MAC scope. An unfamiliar operator is
# surfaced in the report so a new fused kernel cannot silently lower the total.
_EXCLUDED_MAC_OPS = frozenset("""
    _adaptive_avg_pool3d _softmax _to_copy _unsafe_index _unsafe_view
    adaptive_avg_pool2d adaptive_avg_pool3d add add_
    arange batch_norm cat chunk clamp clone constant_pad_nd contiguous copy_
    cudnn_batch_norm detach div dropout dropout_ empty expand flatten gelu index
    layer_norm lift_fresh max_pool3d max_pool3d_with_indices mean mul
    native_batch_norm native_layer_norm _native_batch_norm_legit_no_training
    new_zeros pad permute relu relu_ reshape select sigmoid slice softmax split
    squeeze sub t tanh tensor_split to transpose unbind unsqueeze
    upsample_bilinear2d upsample_linear1d view
""".split())


def _binary_einsum_macs(equation, tensors, output):
    """Count explicit, two-input contractions with matching reduced dimensions."""
    expression = equation.replace(" ", "")
    if len(tensors) != 2 or "..." in expression or "->" not in expression:
        return None
    inputs, result = expression.split("->")
    left, right = inputs.split(",")
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        return None
    reduced = (set(left) | set(right)) - set(result)
    if not reduced or not reduced <= (set(left) & set(right)):
        return None
    left_sizes, right_sizes = (dict(zip(labels, tensor.shape))
                               for labels, tensor in zip((left, right), tensors))
    if any(left_sizes[axis] != right_sizes[axis] for axis in reduced):
        return None
    return output.numel() * math.prod(left_sizes[axis] for axis in reduced)


def count_forward_macs(model, adapter, inputs):
    """Observe one forward, counting each operator once at its dispatch boundary."""
    from torch.utils._python_dispatch import TorchDispatchMode

    calls, macs, unsupported = Counter(), Counter(), Counter()

    class ForwardMacMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            # Redispatch without this mode. This preserves the native kernel and
            # avoids counting both a composite operator and its implementation.
            output = func(*args, **kwargs)
            packet = str(func._overloadpacket)
            calls[packet] += 1
            namespace, _, op = packet.partition(".")
            count = None
            if namespace == "aten":
                if op in _EXCLUDED_MAC_OPS:
                    return output
                if op in {"conv1d", "conv2d", "conv3d", "convolution", "_convolution", "_slow_conv2d_forward"}:
                    transposed = op in {"convolution", "_convolution"} and args[6]
                    if not transposed:
                        count = output.numel() * math.prod(args[1].shape[1:])
                elif op == "linear":
                    count = output.numel() * args[1].shape[-1]
                elif op in {"matmul", "mm", "bmm", "mv", "dot"}:
                    count = output.numel() * args[0].shape[-1]
                elif op in {"addmm", "baddbmm", "addmv"}:
                    count = output.numel() * args[1].shape[-1]
                elif op == "einsum":
                    count = _binary_einsum_macs(args[0], args[1], output)
                elif op in {"scaled_dot_product_attention", "_scaled_dot_product_attention_math",
                            "_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_for_cpu",
                            "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention"}:
                    query, key, value = args[:3]
                    tensor = output[0] if isinstance(output, tuple) else output
                    count = math.prod(tensor.shape[:-1]) * key.shape[-2] * (query.shape[-1] + value.shape[-1])
            if count is None:
                unsupported[packet] += 1
            else:
                macs[packet] += count
            return output

    with ForwardMacMode():
        adapter.step(inputs)
    return {"macs_per_call": None if unsupported else sum(macs.values()), "scope": MAC_SCOPE,
            "macs_by_operator": dict(macs), "operator_calls": dict(calls),
            "unsupported_operators": dict(unsupported)}


def cnn_benchmark_cases(project_root, model_names=("c3dcnn", "lstm", "3dcnnlstm"), batch_size=1):
    root = Path(project_root).resolve()
    registry = {
        "c3dcnn": ("C3D", "C3DCNN.pt"),
        "lstm": ("ConvLSTM", "LSTM.pt"),
        "3dcnnlstm": ("3D-CNN-LSTM", "3DCNNLSTM.pt"),
    }
    return [BenchmarkCase(
        name=registry[name][0],
        model_factory=Factory("cnn_lstm_pytorch:build_model", {
            "model_name": name, "freeze_backbone": True, "verbose": False,
        }),
        checkpoint=str(root / "weights" / registry[name][1]),
        input_shape=(batch_size, 3, 16, 112, 112), batch_size=batch_size,
        parameter_selector=Factory("model_benchmark:cnn_active_parameter_names"),
        mac_counter=Factory("model_benchmark:count_forward_macs"),
        metadata={"implementation": "as_is", "model_name": name},
    ) for name in model_names]


def _device(config):
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                          if config.device == "auto" else config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This timer supports CPU and CUDA; other accelerators need their own synchronization adapter")
    return device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and not isinstance(value, torch.Tensor):
        return _json_value(value.item())
    return str(value)


def _input_description(inputs):
    if isinstance(inputs, torch.Tensor):
        return {"shape": list(inputs.shape), "dtype": str(inputs.dtype), "device": str(inputs.device)}
    if isinstance(inputs, dict):
        return {str(k): _input_description(v) for k, v in inputs.items()}
    if isinstance(inputs, (tuple, list)):
        return [_input_description(v) for v in inputs]
    return {"type": type(inputs).__name__}


def _make_inputs(case, config, device, dtype):
    if case.input_factory:
        return case.input_factory.create(device=device, dtype=dtype, seed=config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    low, high = case.input_range
    return (torch.rand(tuple(case.input_shape), generator=generator) * (high - low) + low).to(device=device, dtype=dtype)


def _load_checkpoint(model, path):
    # Same trusted project checkpoint format as the existing training code, but
    # always deserialize on CPU so no duplicate GPU state_dict is retained.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    model.load_state_dict(state, strict=True)
    metadata = {key: _json_value(payload[key]) for key in ["model_class", "class_names", "epoch"]
                if isinstance(payload, dict) and key in payload}
    del state, payload
    return metadata


@contextlib.contextmanager
def _execution_settings(config, device):
    old = (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        torch.set_num_threads(config.cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.benchmark = False
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(config.seed)
            yield
    finally:
        torch.set_num_threads(old[0])
        torch.backends.cuda.matmul.allow_tf32 = old[1]
        torch.backends.cudnn.allow_tf32 = old[2]
        torch.backends.cudnn.benchmark = old[3]


def _environment(config, device):
    result = {"python": sys.version.split()[0], "pytorch": str(torch.__version__),
              "cuda_build": torch.version.cuda, "device": str(device),
              "platform": platform.platform(), "cpu": platform.processor(),
              "cpu_threads": config.cpu_threads, "dtype": config.dtype,
              "tf32": config.allow_tf32, "amp": False, "cudnn_benchmark": False,
              "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, gpu_total_bytes=props.total_memory, cudnn=torch.backends.cudnn.version())
    return result


def _base_result(case, config):
    return {"name": case.name, "status": "pending", "weights_status": None,
            "mode": case.mode, "state_policy": case.state_policy,
            "comparison_group": case.comparison_group, "batch_size": case.batch_size,
            "frames_per_call": case.frames_per_call, "input_description": case.input_description,
            "checkpoint": case.checkpoint, "case": asdict(case), "config": asdict(config)}


def _warmup(adapter, inputs, case, config):
    adapter.reset()
    for _ in range(config.warmup):
        if case.state_policy == "reset_each_call":
            adapter.reset()
        adapter.step(inputs)


def benchmark_case(case, config):
    """Run one owned model instance. Usually called by the isolated worker.

    Custom factories create CPU models. Inputs are already device-ready; adapter
    reset is outside timing. Carry mode measures a continuous stream within each
    trial and starts a new stream for each trial and measurement phase.
    """
    result = _base_result(case, config)
    path = Path(case.checkpoint).resolve() if case.checkpoint else None
    if (path is None or not path.is_file()) and not case.factory_weights_description and not config.allow_untrained:
        result.update(status="missing_checkpoint", weights_status="missing",
                      error="Supply a checkpoint, or explicitly set allow_untrained=True for architecture-only profiling.")
        return result
    device = _device(config)
    model = adapter = inputs = None
    try:
        with _execution_settings(config, device), torch.inference_mode():
            result["environment"] = _environment(config, device)
            model = case.model_factory.create()
            if not isinstance(model, nn.Module):
                raise TypeError("model_factory must return torch.nn.Module")
            if any(p.device.type != "cpu" for p in [*model.parameters(), *model.buffers()]):
                raise ValueError("model_factory must construct the model on CPU")
            if path is not None and path.is_file():
                loader = case.checkpoint_loader or Factory("model_benchmark:_load_checkpoint")
                result["checkpoint_metadata"] = _json_value(loader.create(model=model, path=path))
                result.update(checkpoint=str(path), weights_status="checkpoint_loaded")
                if config.hash_checkpoints:
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    result["checkpoint_sha256"] = digest.hexdigest()
            elif case.factory_weights_description:
                result.update(weights_status="factory_declared", factory_weights_description=case.factory_weights_description)
            else:
                result["weights_status"] = "initialized_untrained"

            parameters = dict(model.named_parameters())
            result["registered_parameters"] = sum(p.numel() for p in parameters.values())
            result["trainable_parameters"] = sum(p.numel() for p in parameters.values() if p.requires_grad)
            result["active_parameters"] = None
            if case.parameter_selector:
                names = set(case.parameter_selector.create(model=model))
                unknown = names - parameters.keys()
                if unknown:
                    raise ValueError(f"Unknown selected parameters: {sorted(unknown)}")
                result["active_parameters"] = sum(parameters[name].numel() for name in names)
            del parameters
            dtype = getattr(torch, config.dtype)
            model.to(device=device, dtype=dtype).eval()
            inputs = _make_inputs(case, config, device, dtype)
            result["inputs"] = _input_description(inputs)
            adapter = case.adapter_factory.create(model=model)
            if not callable(getattr(adapter, "reset", None)) or not callable(getattr(adapter, "step", None)):
                raise TypeError("adapter must expose reset() and step(inputs)")

            result["macs_per_call"] = None
            result["compute_scope"] = "not configured; no total FLOP claim"
            if case.mac_counter:
                _warmup(adapter, inputs, case, config)
                if case.state_policy == "reset_each_call":
                    adapter.reset()
                compute = case.mac_counter.create(model=model, adapter=adapter, inputs=inputs)
                value = compute.get("macs_per_call")
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError("MAC count must be finite and nonnegative")
                result.update(macs_per_call=value, compute_scope=compute["scope"], compute_details=_json_value(compute))

            samples = []
            for _ in range(config.trials):
                _warmup(adapter, inputs, case, config)
                _sync(device)
                for _ in range(config.iterations):
                    if case.state_policy == "reset_each_call":
                        adapter.reset()
                    _sync(device)
                    start = time.perf_counter()
                    adapter.step(inputs)
                    _sync(device)
                    samples.append((time.perf_counter() - start) * 1000)
            quartiles = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [samples[0]] * 3
            median = statistics.median(samples)
            result.update(latency_ms_median=median, latency_ms_q1=quartiles[0], latency_ms_q3=quartiles[2],
                          latency_ms_mean=statistics.mean(samples), latency_ms_samples=samples,
                          input_frames_per_second=case.batch_size * case.frames_per_call * 1000 / median,
                          peak_allocated_gpu_bytes=None)
            if device.type == "cuda":
                _warmup(adapter, inputs, case, config)
                if case.state_policy == "reset_each_call":
                    adapter.reset()
                _sync(device)
                torch.cuda.reset_peak_memory_stats(device)
                result["baseline_allocated_gpu_bytes"] = torch.cuda.memory_allocated(device)
                adapter.step(inputs)
                _sync(device)
                result["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated(device)
            result["status"] = "ok"
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Drop references before empty_cache. The worker then exits entirely.
        adapter = inputs = model = None
        gc.collect()
        if device.type == "cuda":
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
    return _json_value(result)


def run_benchmarks(cases, config=None, isolated=True):
    """Run cases independently; preserve missing/failed cases as explicit rows.

    Importable factories let subprocesses support adapters defined in project
    modules, including streaming caches and structured/multimodal inputs. Notebook
    closures are intentionally not serialized or executed as arbitrary source.
    """
    config = config or BenchmarkConfig()
    cases = list(cases)
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("Benchmark case names must be unique")
    results = []
    for case in cases:
        if not isolated:
            # Keep factory prints out of notebook tables / JSON protocols.
            with contextlib.redirect_stdout(sys.stderr):
                result = benchmark_case(case, config)
            result["isolation"] = "in_process"
        else:
            payload = json.dumps({"case": asdict(case), "config": asdict(config)})
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            paths = [str(Path(__file__).resolve().parent), *[str(p) for p in sys.path if p]]
            env["PYTHONPATH"] = os.pathsep.join(paths)
            try:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                         input=payload, capture_output=True, text=True, encoding="utf-8",
                                         env=env, timeout=config.timeout_seconds)
                if process.returncode:
                    raise RuntimeError(process.stderr[-4000:] or f"Worker exited {process.returncode}")
                result = json.loads(process.stdout)
                result["worker_log"] = process.stderr[-4000:]
            except (subprocess.SubprocessError, ValueError, RuntimeError) as exc:
                result = _base_result(case, config)
                result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            result["isolation"] = "subprocess"
        results.append(result)
    return results


def results_to_dataframe(results):
    """Compact table; metadata retains raw samples and full per-case conventions."""
    import pandas as pd

    def millions(row, key):
        return round(row[key] / 1e6, 3) if row.get(key) is not None else None

    def input_signature(row):
        def describe(value):
            if isinstance(value, dict):
                if "shape" in value and "dtype" in value:
                    return {"shape": value["shape"], "dtype": value["dtype"]}
                return {key: describe(item) for key, item in value.items()}
            if isinstance(value, list):
                return [describe(item) for item in value]
            return value
        if "inputs" in row:
            return describe(row["inputs"])
        if row["case"].get("input_factory"):
            return {"input_factory": row["case"]["input_factory"]}
        return {"shape": list(row["case"]["input_shape"]), "dtype": "torch." + row["config"]["dtype"]}

    rows = []
    for row in results:
        latency = (f"{row['latency_ms_median']:.3f} [{row['latency_ms_q1']:.3f}, {row['latency_ms_q3']:.3f}]"
                   if row.get("latency_ms_median") is not None else "not measured")
        rows.append({"Model": row["name"], "Status": row["status"], "Weights": row.get("weights_status"),
                     "Workload": f"{row['mode']}/{row['state_policy']}; B={row['batch_size']}; new T={row['frames_per_call']}",
                     "Active params (M)": millions(row, "active_parameters"),
                     "Trainable params (M)": millions(row, "trainable_parameters"),
                     "Registered params (M)": millions(row, "registered_parameters"),
                     "GMAC/call (declared scope)": round(row["macs_per_call"] / 1e9, 3) if row.get("macs_per_call") is not None else None,
                     "Latency median [Q1,Q3] (ms/call)": latency,
                     "Peak allocated GPU (MiB)": round(row["peak_allocated_gpu_bytes"] / 1024**2, 2)
                     if row.get("peak_allocated_gpu_bytes") is not None else None})
    # Heterogeneous models must not silently look like a matched-input comparison.
    if len({row["comparison_group"] for row in results}) > 1:
        for table_row, raw in zip(rows, results):
            table_row["Comparison group"] = raw["comparison_group"]
    if len({json.dumps(input_signature(row), sort_keys=True) for row in results}) > 1:
        for table_row, raw in zip(rows, results):
            table_row["Input specification"] = json.dumps(input_signature(raw))
    return pd.DataFrame(rows)


def export_benchmark_report(results, output_dir, stem="pytorch_efficiency", table=None):
    """Export the table and reproducibility record. No tabulate/LaTeX dependency."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    frame = results_to_dataframe(results) if table is None else table.copy()
    if len(frame) != len(results):
        raise ValueError("Keep one table row per benchmark case, including missing/error cases")
    table = frame.astype(object).where(frame.notna(), "not measured")
    columns = list(table.columns)
    rows = table.values.tolist()
    csv_path, json_path = folder / f"{stem}.csv", folder / f"{stem}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"schema_version": 1, "results": results}, indent=2, allow_nan=False), encoding="utf-8")
    caption = ("Forward-only inference excludes loading, decoding, external preprocessing, host-to-device input transfer, "
               "and state reset. Median [Q1,Q3] over repeated calls. Peak GPU memory includes model, inputs and state. "
               "Registered counts cover the stored model parameters. Active counts use a case-specific selector. "
               "MAC counts cover only the declared operators, not total FLOPs. Compare matching input settings "
               "and comparison_group values; streaming carry and independent-clip workloads are distinct.")
    markdown_path = folder / f"{stem}.md"
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(escape, columns)) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(map(escape, row)) + " |" for row in rows]
    settings = [f"- {r['name']}: group={r['comparison_group']}; inputs={json.dumps(r.get('inputs'))}; "
                f"compute_scope={r.get('compute_scope', 'not measured')}; "
                f"environment={json.dumps(r.get('environment', {}))}" for r in results]
    markdown_path.write_text("\n".join(lines) + "\n\n" + caption + "\n\n" + "\n".join(settings) + "\n", encoding="utf-8")
    latex_path = folder / f"{stem}.tex"
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    latex = lambda value: "".join(replacements.get(char, char) for char in str(value))
    tex = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline", " & ".join(map(latex, columns)) + r" \\"]
    tex += [" & ".join(map(latex, row)) + r" \\" for row in rows]
    tex += [r"\hline", r"\end{tabular}", "% " + caption]
    latex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path, "latex": latex_path}


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        parser.print_help()
        return
    payload = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = benchmark_case(BenchmarkCase.from_dict(payload["case"]), BenchmarkConfig(**payload["config"]))
    sys.stdout.write(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    _main()
