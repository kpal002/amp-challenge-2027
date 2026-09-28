# AMP Challenge 2027 — submission

Research extension: [challenger-guided retrospective experiment](docs/CHALLENGER_EXPERIMENT.md)
and [resumable Colab notebook](notebooks/challenger_colab.ipynb). This evaluates
prediction-error warnings on held-out peptide families; it does not change the
submission pipeline or establish activity of newly generated peptides.
See the [initial development results](docs/CHALLENGER_RESULTS.md) for measured
comparisons against representation-matched controls and their limitations.

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
data supports. That is a measured claim, not an assumption: a 10.69M-parameter
variant (384-dim, 6 layers) was trained and rejected.

| | 1.8M (shipped) | 10.69M (rejected) | held-out AMPs |
|---|---|---|---|
| best validation loss | 1.9393 | **1.8722** | — |
| train/val gap at best epoch | 0.16 | 0.31 | — |
| library FBD ↓ | **0.432** | 0.660 | 0.229 |
| library MMD ↓ | **0.300** | 1.496 | 0.091 |
| Precision / Recall | **0.919 / 0.899** | 0.912 / 0.880 | 0.945 / 0.942 |
| near-duplicates dropped | **5,276** | 6,125 | — |
| generation runtime | **138 s** | 359 s | — |

The larger model wins on validation perplexity and loses on every metric that
matters. Its validation loss bottomed out at epoch 9 and then rose to 2.48 by
epoch 26 while training loss fell to 0.85 — the extra capacity went into
reproducing the 39k-sequence training corpus rather than modelling the
distribution, which shows up as 16% more near-duplicates and a 5x worse MMD.
Perplexity is not the objective.

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

Four filters apply on top of the score:

- **Synthesizability** — no cysteine. Submitted peptides are linear, unmodified
  and free-terminus, with no controlled oxidation step, so a candidate with two or
  more cysteines is not one molecule but an undefined mixture of disulfide
  isomers, and an odd count leaves a free thiol that oxidises in the plate. Either
  way the assay measures something whose identity is not defined.

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
- **Stratified selection** — the list is built by taking the best candidate from
  each of 64 physicochemical strata in turn (quartiles over length, net charge and
  hydrophobic moment), drawing from the top 4,000 candidates by score. Bin edges
  are library-wide: deriving them from the selection pool instead was tried and
  measured worse on four of five categories.

### Why cysteine is excluded

This was found late and is the most consequential correctness fix in the
repository. The generated library is 26.3% cysteine-containing and 12.6%
multi-cysteine, closely matching the reference database (26.4% / 13.3%). The
*selected* top-100, before this filter, was **46% and 40%** — one candidate carried
seven cysteines, and twenty carried an odd count. Roughly 10 of the 25 peptides
drawn for assay would have been undefined mixtures.

The selection stage caused it, not the generator. Cysteine-rich AMPs are among the
most potent entries in GRAMPA, but they are potent *as folded, disulfide-bonded
molecules*; the potency model learned cysteine as a positive signal from them and
ranked accordingly. This is the same error the amidation filter already guards
against on the training side — 43% of GRAMPA is C-terminally amidated and was
dropped for exactly this reason — applied to a different post-translational
structure that the submission rules equally forbid.

| Gate | library eligible | top-100 pred. MIC | multi-Cys | free thiol |
|---|---|---|---|---|
| none | 50,000 | 0.762 | 40 | 20 |
| Cys ≤ 1 | 43,682 | 0.778 | 0 | 17 |
| **Cys = 0 (shipped)** | 36,836 | **0.799** | **0** | **0** |

The shipped gate costs 0.037 log₁₀ units of predicted potency against no gate,
which is 5.6% of the potency model's own clustered-split RMSE of 0.656 — not a
measurable difference by the instrument that measures it. It also moved the
top-100 median net charge to 3.99 against 4.07 for measured-potent AMPs, i.e.
slightly *closer* to the reference distribution. `Cys ≤ 1` would have removed the
isomer problem while leaving 17 free-thiol peptides; excluding cysteine outright
costs 0.021 more and buys an unhedged claim.

A 14–28 residue window was also considered and rejected: it costs four times as
much predicted potency and its lower bound has no synthesis justification, since
short peptides are the easier ones to make.

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

### Why selection is stratified

The envelope bounded *where* the top-100 could sit but not how tightly it
clustered inside those bounds. Measured with `seqme`, score-greedy selection gave
the top-100 **precision 0.97 with recall 0.17** — the signature of a narrow,
homogeneous cluster. Since 25 of the 100 are drawn at random for assay, that
converts the draw into a correlated bet: the sequences differ, but charge and
length barely vary, so they can fail for the same reason.

Stratified selection fixed it, and the effect is measured rather than asserted:

| Metric | score-greedy | stratified, pool 4,000 | **stratified, pool 12,000 (shipped)** | potent refs |
|---|---|---|---|---|
| FBD ↓ | 10.87 | 3.94 | **2.02** | 2.88 |
| MMD ↓ | 72.51 | 17.96 | **2.75** | 9.98 |
| Recall ↑ | 0.17 | 0.86 | 0.85 | 0.88 |
| Diversity ↑ | 0.707 | 0.811 | **0.844** | 0.821 |
| Conformity ↑ | 0.123 | 0.488 | **0.592** | 0.455 |
| top-10 pred MIC | +0.098 | +0.284 | +0.529 | — |


The top-100 now tracks the measured-potent cohort on every axis, and top-10
predicted potency is unchanged (+0.275 → +0.284). Precision falls from 0.97 to
0.86, which is the expected price of coverage.

`--strata-pool` controls the trade-off (default **12,000**). Smaller pools favour
predicted potency, larger ones favour spread. 12,000 captures essentially all the
distributional gain (FBD 2.02 against 1.99 at pool 25,000) while keeping more
predicted potency, and puts the top-100 at median net charge 3.60 against 4.07 for
peptides with verified sub-10 µM activity. Pool 25,000 is half the library, at
which point stratified selection barely differs from sampling the whole library
and the score signal is largely discarded.

### Model performance (out-of-fold)

Two splits are reported. The random split is the conventional number; the
**clustered split** is the honest one. AMP databases are full of homologues and
truncation series, so a random split usually leaves a near-identical peptide in
the training fold. Grouping peptides at Levenshtein ratio >= 0.6 (1,168 groups over
4,121 sequences) and splitting by group estimates performance on unrelated
families.

| Potency model | random split | clustered split |
|---|---|---|
| ridge, Spearman rho | +0.472 | +0.438 |
| MLP, Spearman rho | +0.539 | +0.420 |
| blend (shipped), Spearman rho | +0.557 | **+0.468** |
| blend (shipped), RMSE | 0.606 | **0.656** log10 units |

| AMP-likeness model | AUC |
|---|---|
| logistic regression | 0.690 |
| MLP (shipped) | 0.832 |
| MLP vs shuffled-only negatives | 0.785 |

Scalers are fitted inside each fold. An earlier version fitted one scaler over the
whole dataset before splitting, which leaks test-fold means and variances; the
impact turned out to be negligible here (blend Spearman +0.558 -> +0.557), but the
leak was real and the fix is correct.

Treat **+0.468 Spearman / 0.656 log10 units** as the potency oracle's actual
accuracy. That is roughly a factor of 4.5 in uM. It contributes half the ranking
signal, not all of it.

The AUC against shuffled-only negatives is the meaningful realism number: those
negatives have identical amino-acid composition to the positives, so 0.785
reflects learned sequence *arrangement*. Logistic regression reaches only 0.690
on the same task.

## Generated library characteristics

Medians for the shipped `broad_spectrum` run, against reference cohorts and
decoy controls. Since the aggregation score is withheld, this reproduces the
discrimination it was reportedly tuned for — real AMPs versus decoys — and checks
that the library lands on the AMP side of every axis independently.

| Cohort | pred. log₁₀ MIC | net charge | hydrophobic moment | length |
|---|---|---|---|---|
| generated library | 1.233 | 2.90 | 0.504 | 18 |
| generated top-100 | 0.784 | 3.99 | 0.584 | 18 |
| reference AMPs | 1.150 | 2.99 | 0.560 | 15 |
| potent refs (MIC ≤ 10 µM) | 0.850 | 4.07 | 0.617 | 22 |
| weak refs (MIC ≥ 100 µM) | 1.674 | 2.80 | 0.495 | 15 |


The library tracks the reference AMP distribution on every physicochemical axis
while separating cleanly from composition-matched decoys under the AMP-likeness
model. Reproduce with `uv run python scripts/analyze_library.py`.

### Independent validation with `seqme`

The cohort table above uses our own scorers, which is circular — the models that
ranked the candidates also graded them. `seqme` (the framework the organizers
evaluate with) is independent of this pipeline. Reference database split in half:
one half defines the target distribution, the other is scored as a control, with
no sequence in both. Cohorts and reference are size-matched at n=1000, since
seqme's Precision/Recall require equal sizes.

The **"reference AMPs (held out)"** row is the key control — real AMPs scored
against *other* real AMPs. FBD and MMD have no absolute scale, so that row is
what a near-ideal score looks like on this setup.

| Cohort | FBD ↓ | MMD ↓ | Precision | Recall | Diversity | Conformity |
|---|---|---|---|---|---|---|
| **generated library** | **0.432** | **0.300** | **0.919** | **0.899** | 0.853 | 0.474 |
| reference AMPs (held out) — *ideal* | 0.227 | 0.126 | 0.944 | 0.937 | 0.856 | 0.468 |
| potent refs (MIC ≤ 10 µM) | 1.744 | 10.02 | 0.930 | 0.788 | 0.814 | 0.441 |
| shuffled decoys | 1.416 | 3.79 | 0.898 | 0.855 | 0.856 | 0.476 |
| random peptides | 3.821 | 10.52 | 0.939 | 0.612 | 0.845 | 0.578 |

At the top-100 scale (n=100, matched reference): FBD **2.024**, MMD **2.750**,
precision 0.85, recall 0.85, diversity 0.844, conformity 0.592 — against
2.943 / 10.892 / 0.98 / 0.88 / 0.816 / 0.458 for measured-potent reference AMPs.


Embedder: ESM-2 `t12_35M`. Uniqueness 1.0 and Novelty 1.0 for every generated
cohort. The library sits within ~2× of the ideal control on FBD and MMD while
being 3–9× closer than decoys, with precision, recall, diversity and conformity
all essentially matching the control.

One tension worth naming: *potent* reference AMPs are themselves far from the
overall AMP distribution (FBD 1.744). Distance to "all known AMPs" and distance
to "known potent AMPs" are therefore not the same objective, and the withheld
aggregation score may weight them differently than we have assumed.

| Property | Value |
|---|---|
| Unique sequences | 50,000 / 50,000 |
| Length | min 8, median 18, max 50 |
| Mean pairwise similarity (n=300 subsample) | 0.250 |
| Cysteine-containing peptides in top-100 | 0 / 100 |
| Library novelty vs reference | median 0.667, p95 0.872, max 0.947 |
| Top-100 novelty vs reference | **max 0.750** |
| Top-100 net charge | median 3.99, IQR 2.97 |


Two caveats stated plainly.

The top-100's *predicted* MIC (0.784) is better than the library median (1.233)
but no better than real measured-potent AMPs (0.850), and candidates were selected
on that same prediction. With a clustered-split RMSE of 0.656 log₁₀ units, none of
these gaps is outside model noise. This is not evidence of superiority to known
potent peptides; only the Phase 2 assays can establish that.

Every number in the tables above is emitted by
`uv run python scripts/report_numbers.py` from the committed artifacts, and the
seqme figures by `scripts/evaluate_with_seqme.py`. Earlier revisions of this README
quoted figures copied by hand from different runs and contradicted each other
(top-100 median charge appeared as both 3.99 and 8.99). Regenerate rather than
edit them.

## Known limitations

Stated rather than papered over.

**The potency oracle is weak.** Clustered-split Spearman ρ 0.468 / RMSE 0.656 log₁₀
units is roughly a factor of 4.5 in µM. It carries half the ranking signal.

**Two category labels are proxies.** `mdr` encodes cross-species breadth; no
drug-resistant isolate appears anywhere in the training data, so it is not evidence
of MDR ESKAPE activity. `therapeutic` is essentially a cysteine-free filter with no
measured HC50 supervision at any point, so it is not demonstrated selectivity. The
names come from the competition's award categories, not from the evidence.

**`gram_neg` selects from a narrower band than the other categories.** Its top-100
spans 24 of 64 physicochemical strata, against 59–61 for the others, and its length
IQR is 2.1 against roughly 3 elsewhere. This is the plausibility envelope behaving
correctly rather than a defect: the `gram_neg` conditional is intrinsically
low-charge (library p95 net charge 4.04, against 8.99 for broad-spectrum), and the
envelope floor derived from measured-potent AMPs (net charge ≥ 0, cationic fraction
≥ 0.053, pI ≥ 7.04) removes the low end, leaving candidates spanning ~4.2 charge
units instead of ~7.1. Widening the spread would mean selecting outside the region
where the potency model has evidence, which measured worse. Left as-is.

**No hemolysis data is used anywhere.** The selectivity term is a published
qualitative trend (cationicity favourable, bulk hydrophobicity and long
hydrophobic runs unfavourable), not a fitted model. HC50 enters only in Phase 2.

**The aggregation score is unobservable.** Nothing here is tuned to the
competition metric; the weights were set by reasoning from the published criteria.
Two changes that seemed well-reasoned measured worse and were reverted — see the
`stratum_labels` docstring and the generator-size table above.

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
