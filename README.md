# AMP Challenge 2027 — submission

A category-conditioned peptide language model with a rank-based candidate
selection stage. Generates a 50,000-member antimicrobial peptide library and a
ranked top-100 list per competition category.

## Quick start

```bash
uv sync
uv run generate
```

This writes `generate/library.fasta` (50,000 sequences) and `generate/top.fasta`
(the ranked top 100, best first). Per-category entry points:

```bash
uv run generate_broad_spectrum
uv run generate_gram_pos
uv run generate_gram_neg
uv run generate_mdr
uv run generate_therapeutic
```

`uv run generate` is an alias for `generate_broad_spectrum`. Every entry point
writes into a directory named after itself.

All five categories are produced by one set of weights; the control token selects
the conditional distribution. Each library is 50,000 unique compliant sequences
and each top-100 has maximum reference similarity 0.750. Pairwise overlap between
the five libraries runs from 0.1% to 18.9%, so the conditioning yields genuinely
distinct distributions rather than five relabelled copies:

| Entry point | library median pred. log₁₀ MIC | top-10 median | runtime |
|---|---|---|---|
| `generate_broad_spectrum` | 1.233 | 0.275 | 141 s |
| `generate_gram_pos` | 1.116 | 0.243 | 200 s |
| `generate_gram_neg` | 1.302 | 0.462 | 90 s |
| `generate_mdr` | 1.086 | 0.178 | 202 s |
| `generate_therapeutic` | 1.242 | 0.469 | 151 s |

`mdr` and `gram_pos` draw on the smallest training subsets (1,524 and 1,776
sequences), which sharpens their conditionals and so costs more sampling rounds to
reach 50,000 unique — hence the longer runtimes.

### Reproducibility

The default seed is fixed at 42 and inference is pure NumPy on CPU, so two runs
produce byte-identical files. To check the organizers' criteria locally without
pushing anything:

```bash
uv run python scripts/validate_local.py
```

That runs the entry point twice, byte-compares both FASTA files, and applies
every sequence rule plus the reference-similarity gate.

`--batch-size` is part of the reproducibility contract rather than a tuning
knob: all batches draw from one seeded RNG stream, so changing it changes the
output. The default is what ships.

## Method

### Generator

A 1.8M-parameter decoder-only transformer over the 20-residue alphabet
(4 layers, 6 heads, d_model 192, tanh-GELU, pre-LayerNorm). Each training
example is `[BOS, <category>, residues…, EOS]`, so one set of weights serves all
five categories and the control token selects the conditional distribution at
sampling time.

Peptides are 8–50 residues over a 20-token vocabulary, so this is the scale the
data supports; larger models mostly memorise the reference database.

Sampling uses nucleus sampling (`top_p = 0.95`) with temperatures cycled over
`{0.85, 0.95, 1.00, 1.05, 1.15}` across batches. A single temperature either
collapses diversity (low) or degrades realism (high); a fixed cycle covers more
of the distribution while staying reproducible.

Length validity is enforced in the sampler rather than by rejection: EOS is
masked out until a sequence reaches 8 residues and forced at 50, so every draw
lands in range by construction.

### Inference is NumPy-only, on purpose

PyTorch appears only in the training scripts (`train` dependency group, which uv
does *not* install by default — the `dev` group would be). The
runtime dependencies are NumPy and `levenshtein`. This is a deliberate response
to how submissions are verified: the organizers run `uv sync` and then compare
two runs byte-for-byte, so a CUDA-dependent install and device- or
thread-dependent kernels are pure downside. `training/check_parity.py` verifies
that the NumPy forward pass reproduces the trained PyTorch model's logits.

### Candidate selection

The competition's aggregation score is withheld until Phase 1 closes, so there
is no published objective to optimise against. Ranking therefore combines three
rank-normalised terms rather than chasing a single oracle:

| Term | Source | Weight (broad-spectrum) |
|---|---|---|
| Predicted potency | log₁₀ MIC regressor, blend of MLP and ridge | 0.50 |
| AMP-likeness | classifier vs. composition-matched shuffles | 0.35 |
| Selectivity proxy | charge density penalised by hydrophobicity and hydrophobic runs | 0.15 |

Rank-normalising each term prevents any one from dominating through scale.
The `therapeutic` category reweights toward selectivity (0.45) since it is scored
on the safety window; `mdr` weights potency higher (0.60).

Three filters apply on top of the score:

- **Reference novelty** — required. No top-100 candidate exceeds 0.80 similarity
  to any reference AMP. We enforce 0.75, because the template implements the rule
  as `Levenshtein.ratio` while the competition proposal describes MMseqs2
  alignment identity; the margin keeps candidates valid under either reading.
- **Internal diversity** — no two top-100 candidates exceed 0.80 similarity to
  each other. 25 of the 100 are drawn at random for assay, so a cluster of
  near-identical peptides would turn one design idea into a correlated bet. This
  makes the drawn sample informative about the model rather than about a single
  motif.
- **Plausibility envelope** — candidates must fall within the central 95% range of
  peptides with *measured* MIC ≤ 10 µM, across ten physicochemical descriptors.

### Why the envelope matters

This is the substantive modelling decision in the submission. Ranking on
predicted potency alone produced a top-100 with median net charge **+12.0**,
against **+4.1** for peptides with measured MIC ≤ 10 µM. The potency model never
saw peptides that cationic, so its predictions there were extrapolation rather
than knowledge — and highly cationic, highly amphipathic peptides are
characteristically hemolytic, which the aggregation score penalises through its
comparison against *known potent* peptides.

Constraining to the measured-potent envelope moved the top-100 to median net
charge 9.0 and hydrophobic moment 0.87 (both inside the measured-potent range),
at the cost of nominal predicted potency (top-10 median log₁₀ MIC +0.10 → +0.28).
Giving up a number the oracle could not support is the point, not a regression.

Library sequences are additionally screened against a 0.90 near-duplicate ceiling
(5,276 dropped in the shipped run), since the competition screens for "near-exact
matches" to AMP repositories.

### Model performance (out-of-fold)

| Model | Metric | Value |
|---|---|---|
| Potency, ridge | Spearman ρ | +0.472 |
| Potency, MLP | Spearman ρ | +0.541 |
| Potency, blend (shipped) | Spearman ρ | +0.558 |
| Potency, blend (shipped) | RMSE | 0.605 log₁₀ units |
| AMP-likeness, logistic regression | AUC | 0.690 |
| AMP-likeness, MLP (shipped) | AUC | 0.831 |
| AMP-likeness, MLP vs. shuffled-only negatives | AUC | 0.783 |

The potency RMSE of 0.605 log₁₀ units is roughly a factor of four in µM. This is
a weak oracle and is treated as one — it contributes half the ranking signal, not
all of it.

The AUC against shuffled-only negatives is the meaningful realism number: those
negatives have identical amino-acid composition to the positives, so 0.783
reflects learned sequence *arrangement*. Logistic regression reaches only 0.690
on the same task.

## Generated library characteristics

Medians for the shipped `broad_spectrum` run, against reference cohorts and
decoy controls. Since the aggregation score is withheld, this reproduces the
discrimination it was reportedly tuned for — real AMPs versus decoys — and checks
that the library lands on the AMP side of every axis independently.

| Cohort | pred. log₁₀ MIC | AMP-likeness | net charge | hydrophobic moment | length |
|---|---|---|---|---|---|
| generated library | 1.233 | 0.562 | 2.99 | 0.51 | 18 |
| generated top-100 | 0.475 | 0.997 | 8.99 | 0.90 | 26 |
| reference AMPs | 1.226 | 0.650 | 2.89 | 0.49 | 17 |
| potent refs (MIC ≤ 10 µM) | 0.850 | 0.900 | 4.07 | 0.62 | 22 |
| weak refs (MIC ≥ 100 µM) | 1.674 | 0.731 | 2.80 | 0.50 | 15 |
| shuffled decoys | 1.241 | 0.216 | 2.90 | 0.41 | 18 |
| random peptides | 1.299 | 0.077 | 2.90 | 0.44 | 18 |

The library tracks the reference AMP distribution on every physicochemical axis
while separating cleanly from composition-matched decoys under the AMP-likeness
model. Reproduce with `uv run python scripts/analyze_library.py`.

| Property | Value |
|---|---|
| Unique sequences | 50,000 / 50,000 |
| Length | min 8, median 18, max 50 |
| Mean pairwise similarity (n=300 subsample) | 0.253 |
| Library novelty vs reference | median 0.645, p95 0.889, max 0.960 |
| Top-100 novelty vs reference | median 0.662, p95 0.743, **max 0.750** |

One caveat stated plainly: the top-100's *predicted* MIC (0.475) is better than
that of real measured-potent AMPs (0.850), but candidates were selected on that
same prediction. With an RMSE of 0.605 log₁₀ units, that gap is inside model noise
and partly reflects winner's curse. It is not evidence of superiority to known
potent peptides — only the Phase 2 assays can establish that.

## Training data

All data is public. No proprietary or non-public data was used.

| Source | Use | Rows | License |
|---|---|---|---|
| `data/antibacterial.fasta` | generator corpus, novelty reference | 39,448 AMPs | CC-0 (MarLys, [10.17632/w4hb5grjwb.3](https://doi.org/10.17632/w4hb5grjwb.3)); supplied by the organizers |
| `data/grampa.csv` | MIC regression targets, MDR category definition | 51,345 measurements | [GRAMPA](https://github.com/zswitten/Antimicrobial-Peptides), Witten & Witten |

Derived files, all rebuildable with `uv run python training/build_dataset.py`:

- `data/train_generator.csv` — 77,851 (sequence, category) pairs over 39,655 unique peptides
- `data/train_mic.csv` — 4,121 unique peptides with median log₁₀ MIC
- `data/known_amps.txt` — 39,962 known AMPs, excluded from generated output

### Two data decisions worth flagging

**Only unmodified peptides train the potency model.** 43% of GRAMPA rows carry
C-terminal amidation, which substantially raises measured activity. The
competition forbids terminal modifications, so a model fitted on amidated
measurements would predict the potency of a molecule we are not allowed to
submit. Filtering to unmodified measurements costs training data (4,121
sequences instead of 6,760) and buys a target that matches the deliverable.

**The MDR category is defined from measurements, not labels.** The reference
database's `anti-gram-` and `anti-gram+` labels turn out to be mutually
exclusive, so "active against both" never fires. Instead the MDR set is derived
from GRAMPA: peptides assayed against at least three distinct bacteria with a
median MIC at or below 10 µM (1,524 sequences).

**Novelty exclusion is wider than required.** Only exact matches to
`antibacterial.fasta` are disqualifying, but generated sequences are also
screened against all of GRAMPA, since re-emitting any already-published peptide
would cost novelty points.

## Manual intervention

None on the sequences. No peptide was hand-picked, hand-edited, or removed from
either the library or the top-100 list. Everything below the level of
hyperparameters is produced by `uv run generate`.

Human choices were made at the level of method design: model size, the
temperature schedule, the three ranking terms and their weights, the similarity
margin, and the data filters described above. The ranking weights were set by
reasoning about the published scoring criteria, not tuned against a leaderboard —
the aggregation score is withheld, so there was nothing to tune against.

## Repository layout

```
src/amp_challenge_2027/
  constants.py    competition rules, token vocabulary
  fasta.py        FASTA I/O, parser-compatible with the official validator
  compliance.py   the hard gate: alphabet, length, duplicates, novelty
  features.py     physicochemical and order-sensitive descriptors (NumPy only)
  sampler.py      NumPy transformer inference with a KV cache
  scoring.py      NumPy replay of the trained scorers, ranking policy
  generate.py     entry point
training/
  build_dataset.py    public data -> training corpora
  train_generator.py  trains the language model, exports .npz
  train_ranker.py     trains potency and AMP-likeness models, exports .npz
  check_parity.py     verifies NumPy inference matches trained PyTorch
scripts/
  validate_local.py   runs the organizers' checks against this checkout
tests/                compliance and feature tests
checkpoint/           shipped weights
```

## Retraining from scratch

```bash
uv sync --group train
uv run python training/build_dataset.py
uv run python training/train_generator.py
uv run python training/train_ranker.py
uv run python training/check_parity.py
uv run pytest
```

Training runs on CPU, MPS, or CUDA and takes roughly 15 minutes on an Apple M1
Pro. Training nondeterminism is irrelevant to the submission: what ships is a
fixed checkpoint, and generation from it is deterministic.

## License

MIT — see [LICENSE](LICENSE).
