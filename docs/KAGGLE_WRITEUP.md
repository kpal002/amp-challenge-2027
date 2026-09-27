# Kaggle Writeup — ready to paste

Paste the content below the line into **Writeups → New Writeup** on
<https://www.kaggle.com/competitions/amp-challenge> after pressing **Join
Hackathon**. It covers every minimum and full requirement in one document.

Submit the `generate/` output only — one entry per model, and the five
competition categories are award categories applied to the same peptides.

---

## A category-conditioned peptide language model with distribution-constrained candidate selection

**Repository:** https://github.com/kpal002/amp-challenge-2027 (public, MIT)
**Entry point:** `uv sync && uv run generate` → `generate/library.fasta` (50,000) and `generate/top.fasta` (ranked top 100)

### Abstract

We generate antimicrobial peptide libraries with a 1.8M-parameter decoder-only
transformer over the 20-residue alphabet (4 layers, 6 heads, d_model 192, tanh
GELU, pre-LayerNorm), trained on 39,655 public AMPs. Each training example is
`[BOS, <category>, residues…, EOS]`, so one set of weights serves every
competition category and the control token selects the conditional distribution.
Peptides are 8–50 residues over a 20-token vocabulary, so the model is
deliberately small — see the ablation below, where a 10.69M variant was trained
and rejected.

Validity is enforced inside the sampler rather than by rejection: the
end-of-sequence token is masked until a draw reaches eight residues and forced at
fifty, so every sample is compliant by construction. Sampling uses nucleus
sampling (top_p 0.95) with temperatures cycled over {0.85, 0.95, 1.00, 1.05,
1.15} across batches.

Candidate ranking combines three rank-normalised terms: predicted potency from a
log₁₀ MIC regressor (out-of-fold Spearman ρ 0.558, RMSE 0.605 log₁₀ units), an
AMP-likeness classifier trained against composition-matched shuffles (out-of-fold
AUC 0.783 on that order-only task), and a hemolysis-informed selectivity proxy.
Because the aggregation score is withheld until Phase 1 closes, we did not tune
against a ranking function; we optimised for robustness across the published
metric families independently.

### Top-100 selection procedure

Four filters are applied on top of the composite score.

1. **Reference novelty.** No candidate exceeds 0.75 similarity to any reference
   AMP. The rule states 0.80; we apply a 0.05 margin because the template
   implements it as `Levenshtein.ratio` while the competition proposal specifies
   MMseqs2 alignment identity, and these are not the same measure.
2. **Plausibility envelope.** Candidates must fall within the central 95% range
   of peptides with *measured* MIC ≤ 10 µM across ten physicochemical
   descriptors. Ranking on predicted potency alone produced a top-100 at median
   net charge +12.0 against +4.1 for measured-potent AMPs — outside the oracle's
   training support and in the region where peptides are characteristically
   hemolytic.
3. **Stratified selection.** Candidates are binned into 64 strata (quartiles over
   length, net charge and hydrophobic moment, library-wide edges) and the best
   admissible candidate from each is taken in turn from the top 4,000 by score,
   then re-sorted into rank order. Measured with seqme, score-greedy selection
   inside the envelope gave precision 0.97 with recall 0.17 — a homogeneous
   cluster. Since 25 of the 100 are drawn at random for assay, homogeneity makes
   that draw a correlated bet. Stratifying raised recall to 0.86, diversity to
   0.811 and conformity to 0.488, matching the measured-potent reference cohort
   (0.88 / 0.816 / 0.458), with top-10 predicted potency unchanged.
4. **Internal diversity.** No two candidates exceed 0.80 similarity to each
   other.

### Training data, databases and filters

All data is public; **no proprietary or non-public data was used.**

| Source | Use | License |
|---|---|---|
| `data/antibacterial.fasta` — MarLys reference AMPs (39,448), supplied with the template | generator corpus, novelty reference | CC-0, doi:10.17632/w4hb5grjwb.3 |
| `data/grampa.csv` — GRAMPA MIC compilation (51,345 measurements) | potency targets, MDR category definition | public, github.com/zswitten/Antimicrobial-Peptides |

No AMP database was queried directly; DBAASP, APD, dbAMP and DRAMP enter only as
upstream sources that MarLys and GRAMPA aggregate. Both files are committed so
the corpus is fully disclosed and rebuildable offline.

Two filters worth flagging:

- **Only unmodified peptides train the potency model.** 43% of GRAMPA carries
  C-terminal amidation, which materially raises measured activity, and the
  competition requires free termini. Training on amidated measurements would fit
  an oracle for a molecule we may not submit. This costs training data (4,121
  sequences instead of 6,760) and buys a target that matches the deliverable.
- **The MDR category is defined from measurements, not labels.** The reference
  database's `anti-gram-` and `anti-gram+` labels are mutually exclusive, so
  "active against both" never fires. MDR is instead peptides assayed against ≥3
  distinct bacteria with median MIC ≤ 10 µM (1,524 sequences).

Generated sequences are additionally screened against a 0.90 near-duplicate
ceiling versus known AMPs (5,276 dropped), since the competition screens for
near-exact matches.

### Manual intervention

**None at the sequence level.** No peptide was hand-picked, hand-edited,
reordered or removed. Both files are produced end-to-end by `uv run generate`
with a fixed default seed of 42.

Human judgement applied only to method design: model size, temperature schedule,
the three ranking terms and their weights, the similarity margin, envelope
percentiles and the data filters above. None were tuned against a leaderboard —
the aggregation score is withheld, so there was nothing to tune against.

### Reproducibility

Inference is pure NumPy against committed weights; PyTorch appears only in
training. `uv sync` installs four packages and no CUDA wheels, and generation has
no device- or thread-dependent behaviour. `training/check_parity.py` verifies the
NumPy forward pass reproduces the trained PyTorch logits (max difference 7.6e-6,
identical argmax) and the NumPy scorers reproduce scikit-learn to ~1e-8.

We ran the organizers' own `scripts/verify_submission.py` against the live public
repository: **All checks passed. Submission is valid!** Two consecutive runs
produce byte-identical `library.fasta` and `top.fasta`. 38 tests pass.

### Independent evaluation with seqme

Reference database split in half — one half the target distribution, the other a
held-out control, no overlap. Cohorts size-matched at n=1000.

| Cohort | FBD ↓ | MMD ↓ | Precision | Recall | Conformity |
|---|---|---|---|---|---|
| **generated library** | **0.432** | **0.300** | **0.919** | **0.899** | 0.474 |
| held-out real AMPs (ideal) | 0.227 | 0.126 | 0.944 | 0.937 | 0.468 |
| shuffled decoys | 1.416 | 3.79 | 0.898 | 0.855 | 0.476 |
| random peptides | 3.821 | 10.52 | 0.939 | 0.612 | 0.578 |

Library uniqueness 50,000/50,000; novelty 1.0; mean pairwise similarity 0.253;
top-100 maximum reference similarity 0.750.

### Ablation: a larger model was trained and rejected

| | 1.8M (shipped) | 10.69M (rejected) |
|---|---|---|
| best validation loss | 1.9393 | **1.8722** |
| train/val gap at best epoch | 0.16 | 0.31 |
| library FBD ↓ | **0.432** | 0.660 |
| library MMD ↓ | **0.300** | 1.496 |
| near-duplicates dropped | **5,276** | 6,125 |

The larger model wins on validation perplexity and loses on every metric that
matters. Its validation loss bottomed out at epoch 9 and rose to 2.48 by epoch 26
while training loss fell to 0.85: the capacity went into reproducing the
39k-sequence corpus rather than modelling the distribution. Reproducible via
`scripts/compare_generators.py`.

### Honest caveats

- The potency oracle is weak: RMSE 0.605 log₁₀ units is roughly a factor of four
  in µM. It contributes half the ranking signal, not all of it.
- The top-100's *predicted* MIC (0.475) is better than that of real
  measured-potent AMPs (0.850), but candidates were selected on that same
  prediction. Within the model's noise, this is partly winner's curse and is not
  evidence of superiority to known potent peptides.
- `therapeutic` and `gram_pos` top-100 lists remain more concentrated than
  `broad_spectrum` (FBD 7.33 vs 3.94). We did not ship an unvalidated fix.

### AI assistance disclosure

This submission was developed with the assistance of Claude (Anthropic), used as
a coding and analysis assistant for implementation, data inspection and
validation. All method decisions and the interpretation of results were reviewed
and directed by the human author. The peptides are the output of the trained
model described above, not of a language assistant.
