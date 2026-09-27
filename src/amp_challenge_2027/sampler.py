"""Pure-NumPy inference for the peptide language model.

Mirrors the training architecture exactly (pre-LN decoder-only transformer with
the tanh GELU). Keeping inference free of PyTorch means `uv sync` pulls only
NumPy, and generation has no device- or thread-dependent behaviour to make the
organizers' two-run byte comparison flaky.

All arithmetic is float32 in a fixed operation order, and sampling draws from a
single seeded Generator, so output depends on the seed alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import (
    AA_IDS,
    BLOCK_SIZE,
    BOS_ID,
    EOS_ID,
    MAX_LENGTH,
    MIN_LENGTH,
    VOCAB_SIZE,
    control_id,
    decode,
)

SQRT_2_OVER_PI = np.float32(0.7978845608028654)


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh-approximated GELU, matching torch's ``approximate='tanh'``."""
    inner = SQRT_2_OVER_PI * (x + np.float32(0.044715) * x * x * x)
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + np.float32(1e-5)) * g + b


class PeptideSampler:
    """Autoregressive sampler with a per-layer key/value cache."""

    def __init__(self, weights_path: Path) -> None:
        z = np.load(weights_path)
        self.w = {k: z[k].astype(np.float32) for k in z.files if k != "meta"}
        d_model, n_layers, n_heads, d_ff, block = (int(v) for v in z["meta"])
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.block_size = block
        self.scale = np.float32(1.0 / np.sqrt(self.d_head))

    # -- single decode step -------------------------------------------------
    def _step(
        self,
        token: np.ndarray,          # (B,) int
        t: int,                     # absolute position of `token`
        kcache: list[np.ndarray],   # each (B, H, block, d_head)
        vcache: list[np.ndarray],
    ) -> np.ndarray:
        w = self.w
        B = token.shape[0]
        H, dh = self.n_heads, self.d_head

        x = w["tok_emb"][token] + w["pos_emb"][t]      # (B, D)

        for i in range(self.n_layers):
            p = f"l{i}_"
            h = layer_norm(x, w[p + "ln1_g"], w[p + "ln1_b"])

            q = (h @ w[p + "q_w"] + w[p + "q_b"]).reshape(B, H, dh)
            k = (h @ w[p + "k_w"] + w[p + "k_b"]).reshape(B, H, dh)
            v = (h @ w[p + "v_w"] + w[p + "v_b"]).reshape(B, H, dh)

            kcache[i][:, :, t, :] = k
            vcache[i][:, :, t, :] = v

            ks = kcache[i][:, :, : t + 1, :]           # (B, H, t+1, dh)
            vs = vcache[i][:, :, : t + 1, :]

            # Causal by construction: only positions 0..t exist in the cache.
            scores = np.einsum("bhd,bhtd->bht", q, ks, optimize=True) * self.scale
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            attn = np.einsum("bht,bhtd->bhd", probs, vs, optimize=True)

            a = attn.reshape(B, self.d_model)
            x = x + (a @ w[p + "o_w"] + w[p + "o_b"])

            h = layer_norm(x, w[p + "ln2_g"], w[p + "ln2_b"])
            h = gelu(h @ w[p + "fc1_w"] + w[p + "fc1_b"])
            x = x + (h @ w[p + "fc2_w"] + w[p + "fc2_b"])

        x = layer_norm(x, w["ln_f_g"], w["ln_f_b"])
        return x @ w["head_w"]                          # (B, V)

    # -- sampling -----------------------------------------------------------
    def sample(
        self,
        n: int,
        category: str,
        rng: np.random.Generator,
        temperature: float = 1.0,
        top_p: float = 0.95,
        batch_size: int = 2048,
    ) -> list[str]:
        """Sample `n` peptides. Returns decoded strings (may include invalid ones)."""
        out: list[str] = []
        ctrl = control_id(category)
        for start in range(0, n, batch_size):
            B = min(batch_size, n - start)
            out.extend(self._sample_batch(B, ctrl, rng, temperature, top_p))
        return out

    def _sample_batch(
        self,
        B: int,
        ctrl: int,
        rng: np.random.Generator,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        H, dh = self.n_heads, self.d_head
        kcache = [np.zeros((B, H, self.block_size, dh), dtype=np.float32) for _ in range(self.n_layers)]
        vcache = [np.zeros((B, H, self.block_size, dh), dtype=np.float32) for _ in range(self.n_layers)]

        tokens = np.full((B, self.block_size), EOS_ID, dtype=np.int64)
        alive = np.ones(B, dtype=bool)
        lengths = np.zeros(B, dtype=np.int64)

        # Prime with BOS then the category control token; neither is sampled.
        self._step(np.full(B, BOS_ID, dtype=np.int64), 0, kcache, vcache)
        logits = self._step(np.full(B, ctrl, dtype=np.int64), 1, kcache, vcache)

        allowed = np.array(AA_IDS, dtype=np.int64)
        # MAX_LENGTH + 1 steps: residues may occupy steps 0..MAX_LENGTH-1, and the
        # final step forces EOS, so length 50 stays reachable.
        for step in range(MAX_LENGTH + 1):
            pos = step + 2                      # absolute position of this token
            mask = np.full(VOCAB_SIZE, -np.inf, dtype=np.float32)
            mask[allowed] = 0.0
            # EOS only becomes available once the minimum length is reached, and
            # is forced at the maximum, so every sequence lands in [8, 50].
            if lengths.max() >= MIN_LENGTH:
                mask[EOS_ID] = 0.0

            step_logits = logits + mask
            if step == MAX_LENGTH:
                step_logits = np.full_like(step_logits, -np.inf)
                step_logits[:, EOS_ID] = 0.0
            else:
                # Per-row: forbid EOS for rows still shorter than the minimum.
                too_short = lengths < MIN_LENGTH
                if too_short.any():
                    step_logits[too_short, EOS_ID] = -np.inf

            nxt = self._sample_tokens(step_logits, rng, temperature, top_p)
            nxt = np.where(alive, nxt, EOS_ID)

            tokens[:, step] = nxt
            newly_done = alive & (nxt == EOS_ID)
            lengths += (alive & ~newly_done).astype(np.int64)
            alive = alive & ~newly_done

            if not alive.any():
                break
            logits = self._step(nxt, pos, kcache, vcache)

        return [decode(row.tolist()) for row in tokens]

    @staticmethod
    def _sample_tokens(
        logits: np.ndarray,
        rng: np.random.Generator,
        temperature: float,
        top_p: float,
    ) -> np.ndarray:
        """Nucleus sampling. Deterministic given `rng` state, shapes and dtype."""
        z = logits.astype(np.float32) / np.float32(max(temperature, 1e-6))
        z -= z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=-1, keepdims=True)

        if top_p < 1.0:
            # Stable sort keeps ties in a fixed order across runs.
            order = np.argsort(-p, axis=-1, kind="stable")
            ordered = np.take_along_axis(p, order, axis=-1)
            cum = np.cumsum(ordered, axis=-1)
            # Keep the smallest prefix whose mass reaches top_p (always >= 1 token).
            keep = cum - ordered < top_p
            ordered = np.where(keep, ordered, 0.0)
            ordered /= ordered.sum(axis=-1, keepdims=True)
            p = np.zeros_like(p)
            np.put_along_axis(p, order, ordered, axis=-1)

        # Inverse-CDF draw: one uniform per row, so the number of random values
        # consumed is independent of the distribution.
        u = rng.random(p.shape[0], dtype=np.float32)[:, None]
        return (np.cumsum(p, axis=-1) < u).sum(axis=-1).clip(0, p.shape[-1] - 1)
