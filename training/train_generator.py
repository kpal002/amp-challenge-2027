"""Train the category-conditioned peptide language model.

A small decoder-only transformer over the 20-residue alphabet. Sequences are
short (8-50 residues) and the corpus is ~78k examples, so a ~3M parameter model
trained on CPU/MPS in minutes is the right scale; anything larger mostly
memorises the reference database.

Each example is ``[BOS, <category>, residues..., EOS]``. Conditioning on a
control token means one set of weights serves all five competition categories.

Training may run on MPS or CUDA — nondeterminism here is irrelevant, because what
ships is a fixed checkpoint and inference is pure NumPy on CPU.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amp_challenge_2027.constants import (  # noqa: E402
    BLOCK_SIZE,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    control_id,
    encode,
)

DATA = ROOT / "data"
CHECKPOINT = ROOT / "checkpoint"


class Config:
    d_model = 192
    n_layers = 4
    n_heads = 6
    d_ff = 768
    block_size = BLOCK_SIZE
    vocab_size = VOCAB_SIZE
    dropout = 0.1


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.n_heads = cfg.n_heads
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        h = self.ln1(x)
        q = self.q(h).view(B, T, H, D // H).transpose(1, 2)
        k = self.k(h).view(B, T, H, D // H).transpose(1, 2)
        v = self.v(h).view(B, T, H, D // H).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.o(a))
        h = self.ln2(x)
        # The tanh approximation is used deliberately: it is exactly expressible
        # in NumPy, so the shipped inference path computes the same function that
        # was trained. The default erf form would need SciPy.
        x = x + self.drop(self.fc2(F.gelu(self.fc1(h), approximate="tanh")))
        return x


class PeptideLM(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def build_tensors(df: pd.DataFrame) -> torch.Tensor:
    rows = np.full((len(df), BLOCK_SIZE), PAD_ID, dtype=np.int64)
    for i, (seq, category) in enumerate(zip(df["sequence"], df["category"])):
        ids = [BOS_ID, control_id(category), *encode(seq), EOS_ID]
        rows[i, : len(ids)] = ids
    return torch.from_numpy(rows)


def export_npz(model: PeptideLM, path: Path) -> None:
    """Flatten weights into a plain .npz for the NumPy inference path."""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    out: dict[str, np.ndarray] = {
        "tok_emb": sd["tok.weight"],
        "pos_emb": sd["pos.weight"],
        "ln_f_g": sd["ln_f.weight"],
        "ln_f_b": sd["ln_f.bias"],
        # Store the head transposed so NumPy inference is a plain x @ W.
        "head_w": sd["head.weight"].T,
    }
    for i in range(model.cfg.n_layers):
        p = f"blocks.{i}."
        out[f"l{i}_ln1_g"] = sd[p + "ln1.weight"]
        out[f"l{i}_ln1_b"] = sd[p + "ln1.bias"]
        out[f"l{i}_ln2_g"] = sd[p + "ln2.weight"]
        out[f"l{i}_ln2_b"] = sd[p + "ln2.bias"]
        for name in ("q", "k", "v", "o"):
            out[f"l{i}_{name}_w"] = sd[p + f"{name}.weight"].T
            out[f"l{i}_{name}_b"] = sd[p + f"{name}.bias"]
        out[f"l{i}_fc1_w"] = sd[p + "fc1.weight"].T
        out[f"l{i}_fc1_b"] = sd[p + "fc1.bias"]
        out[f"l{i}_fc2_w"] = sd[p + "fc2.weight"].T
        out[f"l{i}_fc2_b"] = sd[p + "fc2.bias"]

    meta = np.array(
        [model.cfg.d_model, model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_ff, BLOCK_SIZE],
        dtype=np.int64,
    )
    out["meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    df = pd.read_csv(DATA / "train_generator.csv")
    data = build_tensors(df)

    # Split by sequence, not by row: the same peptide appears under several
    # categories, so a row-wise split would leak it across train and validation.
    rng = np.random.default_rng(args.seed)
    unique = np.array(sorted(df["sequence"].unique()))
    held = set(unique[rng.permutation(len(unique))[: int(len(unique) * args.val_frac)]].tolist())
    is_val = torch.from_numpy(df["sequence"].isin(held).to_numpy().copy())
    train, val = data[~is_val], data[is_val]
    print(f"train rows {len(train)} | val rows {len(val)} | held-out peptides {len(held)}")

    cfg = Config()
    model = PeptideLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )

    def evaluate(split: torch.Tensor) -> float:
        model.eval()
        losses, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(split), 512):
                batch = split[i : i + 512].to(device)
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    batch[:, 1:].reshape(-1),
                    ignore_index=PAD_ID,
                )
                losses += loss.item() * len(batch)
                n += len(batch)
        model.train()
        return losses / max(n, 1)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch))
        running = 0.0
        for step in range(steps_per_epoch):
            batch = train[perm[step * args.batch_size : (step + 1) * args.batch_size]].to(device)
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                batch[:, 1:].reshape(-1),
                ignore_index=PAD_ID,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        vl = evaluate(val)
        print(
            f"epoch {epoch:2d}  train {running/steps_per_epoch:.4f}  "
            f"val {vl:.4f}  ppl {math.exp(vl):.2f}  {time.time()-t0:.0f}s"
        )

    config = {k: v for k, v in Config.__dict__.items() if not k.startswith("_")}
    torch.save({"state_dict": model.state_dict(), "config": config}, CHECKPOINT / "generator.pt")
    export_npz(model, CHECKPOINT / "generator.npz")
    print(f"\nwrote {CHECKPOINT/'generator.npz'}")


if __name__ == "__main__":
    main()
