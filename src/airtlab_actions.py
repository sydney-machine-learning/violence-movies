"""Strict AIRTLab action matching shared by preprocessing and label audits."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil


CLASS_NAMES = ("non-violence", "low-level violence", "high-level violence")
NON_VIOLENCE_LABELS = frozenset({
    "handshake", "highfive", "hug", "jump", "walk", "greet",
    "handgestures", "friendly punch",
})
LOW_LEVEL_LABELS = frozenset({"push", "slap", "choke", "fight", "kick", "punch"})
HIGH_LEVEL_LABELS = frozenset({"gunshot", "stab", "club"})
KNOWN_ACTIONS = NON_VIOLENCE_LABELS | LOW_LEVEL_LABELS | HIGH_LEVEL_LABELS
# Preserve the old notebook's vocabulary while using the CSV names canonically.
ACTION_ALIASES = {"shoot": "gunshot", "stifle": "choke"}


def normalize_actions(action_str: str) -> set[str]:
    actions = {" ".join(token.casefold().split()) for token in action_str.split(",")}
    return {ACTION_ALIASES.get(token, token) for token in actions if token}


def _validate_actions(actions: set[str], context: str = "") -> None:
    unknown = actions - KNOWN_ACTIONS
    if unknown:
        raise ValueError(f"Unknown AIRTLab action names {sorted(unknown)} {context}".strip())
    if not actions:
        raise ValueError(f"Missing AIRTLab action names {context}".strip())


def read_csv_map(csv_path: Path | str) -> dict[str, set[str]]:
    csv_path = Path(csv_path)
    mapping = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter=";", skipinitialspace=True), start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if row[0].strip().casefold() == "file":
                continue
            context = f"in {csv_path.name}:{line_number}"
            if len(row) != 2:
                raise ValueError(f"Expected FILE;ACTION CLASSES {context}")
            filename = row[0].strip()
            if Path(filename).name != filename or Path(filename).suffix.casefold() != ".mp4":
                raise ValueError(f"Invalid clip filename {filename!r} {context}")
            if filename in mapping:
                raise ValueError(f"Duplicate clip filename {filename!r} {context}")
            actions = normalize_actions(row[1])
            _validate_actions(actions, f"for {filename} {context}")
            mapping[filename] = actions
    return mapping


def category_for_clip(is_violent_folder: bool, actions: set[str]) -> str:
    actions = normalize_actions(",".join(actions))
    _validate_actions(actions)
    if not is_violent_folder:
        return CLASS_NAMES[0]
    if actions & HIGH_LEVEL_LABELS:
        return CLASS_NAMES[2]
    if actions & LOW_LEVEL_LABELS:
        return CLASS_NAMES[1]
    raise ValueError(f"Violent clip has no recognized violent action: {sorted(actions)}")


@dataclass(frozen=True)
class ClipAssignment:
    source: Path
    destination: Path
    category: str
    camera: str
    actions: frozenset[str]


def plan_airtlab_organization(root: Path | str, output: Path | str) -> list[ClipAssignment]:
    """Validate the entire source inventory before copying or moving any file."""
    root, output = Path(root), Path(output)
    plan = []
    for top, csv_name in (("non-violent", "nonviolent-action-classes.csv"),
                          ("violent", "violent-action-classes.csv")):
        mapping = read_csv_map(root / csv_name)
        for camera in ("cam1", "cam2"):
            folder = root / top / camera
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing AIRTLab source folder: {folder}")
            clips = sorted(folder.glob("*.mp4"))
            actual_names = {p.name for p in clips}
            if actual_names != set(mapping):
                raise ValueError(
                    f"CSV/video mismatch in {folder}: missing CSV rows="
                    f"{sorted(actual_names - set(mapping))}; missing videos={sorted(set(mapping) - actual_names)}"
                )
            for clip in clips:
                category = category_for_clip(top == "violent", mapping[clip.name])
                plan.append(ClipAssignment(clip, output / category / camera / clip.name,
                                           category, camera, frozenset(mapping[clip.name])))
    return plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def organize_airtlab(root: Path | str, output: Path | str, copy_files=True, dry_run=False):
    """Organize validated clips and remove only verified identical stale copies.

    In particular, rerunning after the gunshot correction cannot leave a clip
    in both low/high folders. Different existing content is never overwritten.
    Existing memmaps are not rebuilt here; the notebook validates their labels.
    """
    output = Path(output)
    plan = plan_airtlab_organization(root, output)
    stale = []
    for item in plan:
        candidates = [item.destination]
        if item.category != CLASS_NAMES[0]:
            candidates.extend(output / other / item.camera / item.source.name
                              for other in CLASS_NAMES[1:] if other != item.category)
        existing = [path for path in candidates if path.exists()]
        if existing:
            source_digest = _sha256(item.source)
            for path in existing:
                if _sha256(path) != source_digest:
                    raise ValueError(f"Existing output differs from source; refusing to replace/remove: {path}")
                if path != item.destination:
                    stale.append(path)
    if not dry_run:
        for item in plan:
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            if copy_files:
                if not item.destination.exists():
                    shutil.copy2(item.source, item.destination)
            else:
                shutil.move(str(item.source), str(item.destination))
        for path in stale:
            path.unlink()
    return {"videos": len(plan), "videos_by_class": dict(Counter(p.category for p in plan)),
            "stale_copies": len(stale), "dry_run": dry_run}


def validate_prepared_action_labels(root: Path | str, output: Path | str):
    """Read-only check of cached labels against CSV actions; no frame decoding."""
    import numpy as np

    root, output = Path(root), Path(output)
    maps = {
        False: read_csv_map(root / "nonviolent-action-classes.csv"),
        True: read_csv_map(root / "violent-action-classes.csv"),
    }
    manifest = json.loads((output / "mmap_manifest.json").read_text(encoding="utf-8"))
    expected = {}
    mismatches = []
    for video in manifest["videos"]:
        path = Path(video["path"])
        violent = video["class_name"] != CLASS_NAMES[0]
        if path.name not in maps[violent]:
            raise ValueError(f"Missing CSV actions for prepared video {path}")
        category = category_for_clip(violent, maps[violent][path.name])
        label = CLASS_NAMES.index(category)
        expected[video["video_id"]] = (label, category)
        if video["class_name"] != category or video["label"] != label or path.parent.parent.name != category:
            mismatches.append(str(path))
    labels = np.fromfile(output / "labels.mmap", dtype=np.int8)
    videos = np.fromfile(output / "videos.mmap", dtype=np.int32)
    if len(labels) != manifest["total_chunks"] or len(videos) != len(labels):
        raise ValueError("AIRTLab cached label/video count mismatch")
    for video_id, (label, _) in expected.items():
        if np.any(labels[videos == video_id] != label):
            mismatches.append(f"labels.mmap video ID {video_id}")
    metadata_path = output / "chunk_metadata.jsonl"
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            label, category = expected[row["video_id"]]
            if row["label"] != label or row["class_name"] != category or Path(row["video_path"]).parent.parent.name != category:
                mismatches.append(f"chunk_metadata.jsonl sample {row['sample_index']}")
    if mismatches:
        raise ValueError(
            f"Stale AIRTLab action labels ({len(mismatches)} mismatches). Reorganize clips and set "
            f"REBUILD_AIRTLAB_MMAPS=True; rebuild combined arrays afterward. Examples: {mismatches[:3]}"
        )
    return {"videos": len(expected), "chunks": len(labels),
            "chunk_labels": dict(Counter(int(x) for x in labels)), "unmatched_actions": 0}
