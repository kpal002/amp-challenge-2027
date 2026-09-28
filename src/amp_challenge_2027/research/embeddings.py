"""Resumable, revision-pinned ESM residue-mean embeddings for Colab."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .data import atomic_json, digest, load_prepared

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
DEFAULT_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def read_vectors(path: Path, sequences: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if data["sequences"].tolist() != sequences:
            raise ValueError(f"Embedding sequence/order mismatch: {path}")
        vectors = data["vectors"]
    if vectors.ndim != 2 or len(vectors) != len(sequences) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid embedding matrix: {path}")
    return vectors


def embed(work: Path, model_id: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION,
          batch_size: int = 64, device: str = "auto", local_only: bool = False) -> Path:
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if batch_size < 1 or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a positive batch size and a full 40-character model commit SHA")
    _, split, _ = load_prepared(work)
    sequences = split.sequence.tolist()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = work / "embeddings"
    cache.mkdir(parents=True, exist_ok=True)
    metadata = {"model": model_id, "revision": revision, "sequence_sha256": digest(sequences),
                "pooling": "last hidden state; mean residues; excludes BOS/EOS/PAD",
                "dtype": "float32", "batch_size": batch_size, "device": device,
                "torch": torch.__version__, "transformers": transformers.__version__}
    meta_path = cache / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Embedding configuration changed; use another work directory")
    atomic_json(meta_path, metadata)
    final = cache / "vectors.npz"
    if final.exists():
        read_vectors(final, sequences)
        print(f"Reusing complete embeddings: {final}", flush=True)
        return final
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                               local_files_only=local_only)
    model = AutoModel.from_pretrained(model_id, revision=revision,
                                     local_files_only=local_only).to(device).eval()
    parts = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        path = cache / f"part-{start:06d}.npz"
        if path.exists():
            vectors = read_vectors(path, batch)
        else:
            tokens = tokenizer(batch, padding=True, return_tensors="pt",
                               return_special_tokens_mask=True)
            special = tokens.pop("special_tokens_mask").bool().to(device)
            tokens = {key: value.to(device) for key, value in tokens.items()}
            residue_mask = tokens["attention_mask"].bool() & ~special
            if residue_mask.sum(1).tolist() != [len(s) for s in batch]:
                raise ValueError("Tokenizer does not produce exactly one token per residue")
            with torch.inference_mode():
                hidden = model(**tokens).last_hidden_state
                vectors = ((hidden * residue_mask.unsqueeze(-1)).sum(1)
                           / residue_mask.sum(1, keepdim=True)).float().cpu().numpy()
            atomic_npz(path, sequences=np.array(batch), vectors=vectors)
        parts.append(vectors)
        print(f"Embeddings {min(start + batch_size, len(sequences))}/{len(sequences)}", flush=True)
    atomic_npz(final, sequences=np.array(sequences), vectors=np.concatenate(parts),
               metadata=np.array(json.dumps(metadata, sort_keys=True)))
    return final
