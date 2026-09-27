"""Verify the NumPy inference path reproduces the trained PyTorch model.

The submitted artifact runs on NumPy; the weights were fitted in PyTorch. If the
two forward passes disagree, the shipped model is not the model that was trained
and every validation number above it is meaningless. This checks logits directly
rather than inferring agreement from sample quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amp_challenge_2027.constants import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    control_id,
    encode,
)
from amp_challenge_2027.sampler import PeptideSampler  # noqa: E402

sys.path.insert(0, str(ROOT / "training"))
from train_generator import Config, PeptideLM  # noqa: E402

CHECKPOINT = ROOT / "checkpoint"


def torch_logits(model: PeptideLM, rows: list[list[int]]) -> np.ndarray:
    idx = torch.tensor(rows, dtype=torch.long)
    with torch.no_grad():
        return model(idx).float().numpy()


def numpy_logits(sampler: PeptideSampler, rows: list[list[int]]) -> np.ndarray:
    """Replay the cached decode step by step and collect logits per position."""
    B, T = len(rows), len(rows[0])
    H, dh = sampler.n_heads, sampler.d_head
    kcache = [np.zeros((B, H, sampler.block_size, dh), dtype=np.float32) for _ in range(sampler.n_layers)]
    vcache = [np.zeros((B, H, sampler.block_size, dh), dtype=np.float32) for _ in range(sampler.n_layers)]
    arr = np.array(rows, dtype=np.int64)
    out = np.zeros((B, T, sampler.w["head_w"].shape[1]), dtype=np.float32)
    for t in range(T):
        out[:, t, :] = sampler._step(arr[:, t], t, kcache, vcache)
    return out


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "generator"
    ckpt = torch.load(CHECKPOINT / f"{name}.pt", map_location="cpu", weights_only=False)
    saved = ckpt.get("config", {})
    # Rebuild the exact architecture that was trained rather than the defaults.
    model = PeptideLM(
        Config(
            d_model=saved.get("d_model", 192),
            n_layers=saved.get("n_layers", 4),
            n_heads=saved.get("n_heads", 6),
            d_ff=saved.get("d_ff"),
        )
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sampler = PeptideSampler(CHECKPOINT / f"{name}.npz")

    peptides = [
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "KRWWKWWRR",
        "FLPIIAKVLSGLL",
        "ACDEFGHIKLMNPQRSTVWY",
    ]
    rows = []
    for seq, category in zip(peptides, ("broad_spectrum", "gram_pos", "gram_neg", "mdr")):
        ids = [BOS_ID, control_id(category), *encode(seq), EOS_ID]
        rows.append(ids)
    width = min(len(r) for r in rows)
    rows = [r[:width] for r in rows]

    a = torch_logits(model, rows)
    b = numpy_logits(sampler, rows)

    diff = np.abs(a - b)
    print(f"shape {a.shape}")
    print(f"max abs logit difference : {diff.max():.3e}")
    print(f"mean abs logit difference: {diff.mean():.3e}")

    # float32 accumulation differs in order between torch and NumPy, so exact
    # equality is not expected; agreement to ~1e-3 confirms the same function.
    agree_argmax = (a.argmax(-1) == b.argmax(-1)).mean()
    print(f"argmax agreement        : {agree_argmax:.4f}")

    generator_ok = diff.max() < 1e-2 and agree_argmax == 1.0

    ranker_ok = check_ranker()

    ok = generator_ok and ranker_ok
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    if not ok:
        raise SystemExit(1)


def check_ranker() -> bool:
    """Compare the NumPy scorer against scikit-learn's own stored predictions."""
    from amp_challenge_2027.scoring import Scorer

    sys.path.insert(0, str(ROOT / "training"))
    from train_ranker import PROBES

    path = CHECKPOINT / "ranker.npz"
    stored = np.load(path)
    if "probe_pot" not in stored.files:
        print("\nranker: no stored probe predictions; re-run train_ranker.py")
        return False

    scorer = Scorer(path)
    X = scorer.features(list(PROBES))
    pot = scorer.predict_log_mic(X)
    amp = scorer.predict_amp_likeness(X)

    d_pot = np.abs(pot - stored["probe_pot"]).max()
    d_amp = np.abs(amp - stored["probe_amp"]).max()
    print(f"\nranker potency  max abs difference vs sklearn: {d_pot:.3e}")
    print(f"ranker likeness max abs difference vs sklearn: {d_amp:.3e}")
    return bool(d_pot < 1e-4 and d_amp < 1e-4)


if __name__ == "__main__":
    main()
