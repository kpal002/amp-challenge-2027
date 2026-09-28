"""Provenance-aware labels and transitive sequence-family splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd

from amp_challenge_2027.compliance import is_valid_sequence

SPLITS = ("train", "calibration", "validation", "test")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_flag(series: pd.Series) -> pd.Series:
    """Never let a nonempty string such as 'False' become True via astype(bool)."""
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Missing or unrecognised boolean in {series.name}")
    return mapped.astype(bool)


def load_measurements(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    initial = len(frame)
    frame["sequence"] = frame.sequence.fillna("").astype(str).str.strip().str.upper()
    frame["bacterium"] = frame.bacterium.fillna("").astype(str).str.strip()
    for column in ("is_modified", "has_cterminal_amidation", "has_unusual_modification",
                   "datasource_has_modifications"):
        frame[column] = parse_flag(frame[column])
    keep = frame.sequence.map(is_valid_sequence) & frame.bacterium.ne("")
    keep &= ~(frame.is_modified | frame.has_cterminal_amidation | frame.has_unusual_modification)
    frame = frame.loc[keep].copy()
    # This GRAMPA release stores log10(uM), despite its unit column saying uM.
    # Do NOT log-transform a second time. Check the pinned input hash in manifest.
    frame["log_mic"] = pd.to_numeric(frame.value, errors="raise")
    if not np.isfinite(frame.log_mic).all() or not frame.unit.eq("uM").all():
        raise ValueError("Expected finite GRAMPA log10 MIC values with unit uM")
    frame["confirmed"] = frame.datasource_has_modifications
    frame["strain"] = frame.strain.fillna("unspecified").astype(str)
    if frame.database.isna().any():
        raise ValueError("Missing measurement source")
    # Preserve source membership for source-ablation models. Collapse identical
    # observations across sources only when computing each model's target.
    frame = frame.drop_duplicates(["sequence", "bacterium", "strain", "database",
                                   "log_mic", "confirmed"])
    audit = {"input_rows": initial, "eligible_rows_after_dedup": len(frame),
             "unknown_modification_rows": int((~frame.confirmed).sum()),
             "eligible_sequences": int(frame.sequence.nunique()),
             "target_units": "log10(uM); input value already log transformed"}
    return frame, audit


def aggregate_labels(frame: pd.DataFrame, species: str, *, strict: bool = True,
                     omit_source: str | None = None) -> pd.Series:
    selected = frame[frame.bacterium.eq(species)]
    if strict:
        selected = selected[selected.confirmed]
    if omit_source is not None:
        selected = selected[selected.database.ne(omit_source)]
    selected = selected.drop_duplicates(["sequence", "bacterium", "strain", "log_mic"])
    # Equal weight per observed strain, rather than per database copy or assay.
    # An unspecified strain is one group, not inferred to be a competition strain.
    per_strain = selected.groupby(["sequence", "strain"]).log_mic.median()
    return per_strain.groupby(level="sequence").median().sort_index()


def connected_families(sequences: list[str], threshold: float = 0.7) -> np.ndarray:
    """Exact connected components: every pair >= threshold shares a component.

    Representative-only clustering does not provide that guarantee. Length
    pruning here is exact under Levenshtein.ratio = 2*LCS/(n+m).
    """
    if not 0 < threshold <= 1:
        raise ValueError("Family threshold must be in (0, 1]")
    parent = np.arange(len(sequences))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for i, sequence in enumerate(sequences):
        for j in range(i):
            other = sequences[j]
            if 2 * min(len(sequence), len(other)) / (len(sequence) + len(other)) < threshold:
                continue
            if Levenshtein.ratio(sequence, other) >= threshold:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
    roots = [root(i) for i in range(len(sequences))]
    identifiers = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([identifiers[r] for r in roots])


def assign_splits(groups: np.ndarray, seed: int = 42) -> np.ndarray:
    """Balance component sizes using sequences only; never inspect activity labels."""
    ids, counts = np.unique(groups, return_counts=True)
    if len(ids) < 4:
        raise ValueError("Need at least four sequence families for four partitions")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(ids))
    order = sorted(shuffled, key=lambda i: -counts[i])
    targets = len(groups) * np.array([0.55, 0.15, 0.15, 0.15])
    sizes = np.zeros(4)
    mapping = {}
    for i in order:
        split = int(np.argmax(targets - sizes))
        mapping[ids[i]] = SPLITS[split]
        sizes[split] += counts[i]
    if np.any(sizes == 0):
        raise ValueError("A giant connected family leaves an empty partition")
    return np.array([mapping[g] for g in groups])


def prepare(source: Path, work: Path, threshold: float, seed: int) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    config = {"schema": 1, "source_sha256": file_digest(source),
              "family_threshold": threshold, "seed": seed,
              "split_fractions": [0.55, 0.15, 0.15, 0.15]}
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != config:
            raise ValueError("Dataset/split configuration changed; use a new work directory")
        for name, expected in manifest["files"].items():
            if file_digest(work / name) != expected:
                raise ValueError(f"Prepared artifact changed: {name}")
        return manifest
    frame, audit = load_measurements(source)
    sequences = sorted(frame.sequence.unique())
    print(f"Clustering {len(sequences)} sequences at ratio >= {threshold}", flush=True)
    groups = connected_families(sequences, threshold)
    assignments = assign_splits(groups, seed)
    split_frame = pd.DataFrame({"sequence": sequences, "family": groups, "split": assignments})
    frame.to_csv(work / "measurements.csv", index=False)
    split_frame.to_csv(work / "splits.csv", index=False)
    audit.update({"families": len(set(groups)), "largest_family": int(np.bincount(groups).max()),
                  "split_sizes": split_frame.split.value_counts().to_dict()})
    manifest = {"config": config, "audit": audit,
                "files": {name: file_digest(work / name) for name in ("measurements.csv", "splits.csv")}}
    atomic_json(manifest_path, manifest)
    return manifest


def load_prepared(work: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((work / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if file_digest(work / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    frame = pd.read_csv(work / "measurements.csv")
    frame["confirmed"] = parse_flag(frame.confirmed)
    return frame, pd.read_csv(work / "splits.csv"), manifest
