"""Evaluate the generated library with seqme -- the framework the organizers use.

Not part of the submission pipeline and deliberately not a project dependency.
seqme pulls PyTorch and transformers, which the generation path does not need.
Run it in a separate environment:

    uv venv --python 3.11 /tmp/seqme && \
      VIRTUAL_ENV=/tmp/seqme uv pip install "seqme[esm2,aa_descriptors]"
    /tmp/seqme/bin/python scripts/evaluate_with_seqme.py 1000


Everything up to now validated the library with our own scorers, which is
circular: the same models that ranked the candidates also graded them. seqme is
independent of our pipeline, so this is the first genuinely external check.

The competition's aggregation weights and curated reference sets are withheld, so
absolute leaderboard numbers are not reproducible. What is reproducible is each
metric family individually, and the comparison against decoy controls.

Precision/Recall in seqme require the evaluated cohort and the reference set to
have equal size, so the run is split by cohort size rather than disabling that
check -- mismatched sizes would bias exactly the metrics we care about.

The key control is "reference AMPs (held out)": real AMPs scored against *other*
real AMPs. It shows what a near-ideal score looks like on this setup, which is
the only way to read FBD/MMD numbers that have no absolute scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import seqme
import seqme.metrics as M
import seqme.models as Mo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from amp_challenge_2027.fasta import read_sequences  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(0)


def sub(seqs: list[str], k: int, generator: np.random.Generator) -> list[str]:
    if len(seqs) <= k:
        return list(seqs)
    idx = generator.choice(len(seqs), size=k, replace=False)
    return [seqs[i] for i in idx]


library = read_sequences(REPO / "generate" / "library.fasta")
top = read_sequences(REPO / "generate" / "top.fasta")
reference = read_sequences(REPO / "data" / "antibacterial.fasta")

mic = pd.read_csv(REPO / "data" / "train_mic.csv")
refset = set(reference)
potent = [s for s, v in zip(mic.sequence, mic.log10_mic) if v <= 1.0 and s in refset]

# Split the reference database once: half defines the target distribution, half
# supplies the held-out control. No sequence appears in both.
order = rng.permutation(len(reference))
ref_all = [reference[i] for i in order]
half = len(ref_all) // 2
ref_pool, control_pool = ref_all[:half], ref_all[half:]

embedder = Mo.ESM2(Mo.ESM2Checkpoint.t12_35M, device="cpu", batch_size=64)
descriptors = [Mo.Charge(), Mo.Hydrophobicity(), Mo.HydrophobicMoment(), Mo.Gravy()]


def build_metrics(ref_target: list[str], n_neighbors: int = 5) -> list:
    return [
        M.Uniqueness(),
        M.Diversity(),
        M.Novelty(reference=ref_target),
        M.FBD(reference=ref_target, embedder=embedder),
        M.MMD(reference=ref_target, embedder=embedder),
        M.Precision(n_neighbors=n_neighbors, reference=ref_target, embedder=embedder),
        M.Recall(n_neighbors=n_neighbors, reference=ref_target, embedder=embedder),
        M.ConformityScore(reference=ref_target, predictors=descriptors),
    ]


def run(label: str, size: int, include_top: bool) -> pd.DataFrame:
    g = np.random.default_rng(1)
    ref_target = sub(ref_pool, size, g)

    lib_s = sub(library, size, g)
    shuffled = ["".join(g.permutation(np.array(list(s)))) for s in lib_s]
    pool = np.array(list("".join(sub(ref_pool, 2000, g))))
    random_peptides = ["".join(g.choice(pool, size=len(s), replace=True)) for s in lib_s]

    cohorts: dict[str, list[str]] = {}
    if include_top:
        cohorts["generated top-100"] = list(top)
    cohorts["generated library"] = lib_s
    cohorts["reference AMPs (held out)"] = sub(control_pool, size, g)
    cohorts["potent refs (MIC<=10uM)"] = sub(potent, size, g)
    cohorts["shuffled decoys"] = shuffled
    cohorts["random peptides"] = random_peptides

    print(f"\n### {label}: cohort size {size}, reference target {len(ref_target)}")
    df = seqme.evaluate(cohorts, build_metrics(ref_target), verbose=False)
    return df


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

big = run(f"library-scale (n={N})", N, include_top=False)
print(big.to_string())

small = run("top-100 scale (n=100)", 100, include_top=True)
print(small.to_string())

out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
big.to_csv(out / "seqme_library_scale.csv")
small.to_csv(out / "seqme_top_scale.csv")
print("\nwrote seqme_library_scale.csv and seqme_top_scale.csv")
