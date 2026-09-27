"""Compare two generator checkpoints on the seqme metrics that matter.

Validation perplexity is not the objective: a larger model can lower it by
memorising the training corpus while getting worse on distributional similarity
and novelty. This compares libraries directly.

Usage (in the seqme environment -- see scripts/evaluate_with_seqme.py):
    python scripts/compare_generators.py <candidate-output-dir>

where the candidate directory holds a library.fasta generated with
`uv run generate --weights checkpoint/<candidate>.npz --out-dir <dir>`.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import seqme, seqme.metrics as M, seqme.models as Mo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from amp_challenge_2027.fasta import read_sequences

N = 1000
rng = np.random.default_rng(0)
reference = read_sequences(REPO / "data" / "antibacterial.fasta")
order = rng.permutation(len(reference))
ref_all = [reference[i] for i in order]
half = len(ref_all)//2
ref_pool, control_pool = ref_all[:half], ref_all[half:]

g = np.random.default_rng(1)
def sub(s, k, gen):
    return list(s) if len(s) <= k else [s[i] for i in gen.choice(len(s), size=k, replace=False)]

ref_target = sub(ref_pool, N, g)
embedder = Mo.ESM2(Mo.ESM2Checkpoint.t12_35M, device="cpu", batch_size=64)
metrics = [
    M.Diversity(),
    M.Novelty(reference=ref_target),
    M.FBD(reference=ref_target, embedder=embedder),
    M.MMD(reference=ref_target, embedder=embedder),
    M.Precision(n_neighbors=5, reference=ref_target, embedder=embedder),
    M.Recall(n_neighbors=5, reference=ref_target, embedder=embedder),
    M.ConformityScore(reference=ref_target,
        predictors=[Mo.Charge(), Mo.Hydrophobicity(), Mo.HydrophobicMoment(), Mo.Gravy()]),
]
cohorts = {
    "1.8M (shipped)": sub(read_sequences(REPO/"generate"/"library.fasta"), N, g),
    "10.69M (candidate)": sub(read_sequences(Path(sys.argv[1])/"library.fasta"), N, g),
    "[control] held-out AMPs": sub(control_pool, N, g),
}
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
print(seqme.evaluate(cohorts, metrics, verbose=False).to_string())
