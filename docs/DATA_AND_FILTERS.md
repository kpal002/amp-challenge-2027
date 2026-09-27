# Training data, external databases, and filters

Submitted in satisfaction of the requirement for "a short summary describing
training data, external databases used, and any manual intervention or
computational filters applied."

## 1. Data sources

All data is public. **No proprietary or non-public data was used**, so no data
release obligation arises.

| Source | Provenance | Use |
|---|---|---|
| `data/antibacterial.fasta` | MarLys reference AMP set, supplied with the competition template. CC-0, Mendeley DOI [10.17632/w4hb5grjwb.3](https://doi.org/10.17632/w4hb5grjwb.3). 39,448 sequences, all canonical and all 8–50 residues. | Generator training corpus; novelty reference; exact-match exclusion. |
| `data/grampa.csv` | GRAMPA, Witten & Witten — a compilation of MIC measurements aggregated from APD, DADP, DBAASP, DRAMP, PEP_LIFE and YADAMP. Retrieved from [github.com/zswitten/Antimicrobial-Peptides](https://github.com/zswitten/Antimicrobial-Peptides). 51,345 measurements over 6,760 unique peptides. | Potency regression targets; definition of the MDR category; wider novelty exclusion. |

No AMP database was queried directly. DBAASP, APD, dbAMP and Peptipedia enter
only indirectly, as the upstream sources that MarLys and GRAMPA aggregate.

Both files are committed to the repository so the corpus is fully disclosed and
the pipeline is rebuildable without network access.

## 2. Derived training sets

Rebuildable with `uv run python training/build_dataset.py`.

| File | Contents |
|---|---|
| `data/train_generator.csv` | 77,851 (sequence, category) pairs over 39,655 unique peptides |
| `data/train_mic.csv` | 4,121 unique peptides with median log₁₀ MIC (µM) |
| `data/known_amps.txt` | 39,962 known AMPs, excluded from generated output |

### Filters applied to the potency training set

1. Canonical alphabet and length 8–50 (matching the submission constraints).
2. **Unmodified peptides only** — rows flagged `is_modified`,
   `has_cterminal_amidation` or `has_unusual_modification` are dropped. 43% of
   GRAMPA carries C-terminal amidation, which materially increases measured
   activity, and the competition requires free termini. Keeping amidated
   measurements would fit an oracle for a molecule we are not permitted to
   submit. This reduces the usable set from 6,760 to 4,121 peptides.
3. Replicate measurements aggregated by **median**, not mean: GRAMPA pools
   multiple databases and bacterial strains, so a single sequence can carry
   dozens of values whose spread reflects assay conditions rather than the
   peptide.

### Category definitions

- `broad_spectrum` — every peptide in the reference database (all are antimicrobial).
- `gram_neg` / `gram_pos` — reference database `anti-gram-` / `anti-gram+` activity labels (5,868 / 1,776 sequences).
- `mdr` — **derived from measurements, not labels.** The database's `anti-gram-` and `anti-gram+` labels turn out to be mutually exclusive, so "active against both" never fires and cannot define a cross-spectrum category. Instead: peptides assayed against ≥3 distinct bacteria with median MIC ≤ 10 µM (1,524 sequences).
- `therapeutic` — cysteine-free reference peptides, which cannot form disulfides and are therefore the cleanest examples of the strictly linear, free-terminus peptides this category requires (29,028 sequences).

## 3. Computational filters applied to generated output

Applied in order, inside `uv run generate`. No step involves human inspection.

**Hard compliance gate (library and top-100).** Alphabet restricted to the 20
standard residues; length 8–50; no duplicates; zero exact matches to the
reference database. Length compliance is additionally structural: the sampler
masks the end-of-sequence token until a draw reaches 8 residues and forces it at
50, so out-of-range sequences are never produced in the first place.

**Wider novelty exclusion (library).** Generated sequences are excluded if they
exactly match any of the 39,962 peptides in `known_amps.txt` — a superset of the
disqualifying reference set, since re-emitting any published peptide would cost
novelty for no benefit.

**Near-duplicate screen (library).** Sequences exceeding 0.90 similarity to any
reference AMP are dropped (5,276 dropped in the shipped run). A k-mer index
provides the fast path; sequences sharing no 10-mer with any reference are
accepted without the expensive comparison. This is a quality filter rather than a
compliance gate, so the heuristic fast path is acceptable — it admits a small
residual tail, measured at max 0.960 in the shipped library.

**Reference-similarity gate (top-100).** No candidate exceeds **0.75** similarity
to any reference AMP. The rule states 0.80; we apply a 0.05 margin because the
template implements it as `Levenshtein.ratio` while the competition proposal
specifies MMseqs2 alignment identity, and these are not the same measure.

**Plausibility envelope (top-100).** Candidates must fall within the central 95%
range (p2.5–p97.5) of peptides with *measured* MIC ≤ 10 µM, across ten
descriptors: net charge, charge per residue, hydrophobic moment, hydrophobic
fraction, GRAVY, length, cationic fraction, Boman index, isoelectric point, and
longest hydrophobic run. Without this, ranking on predicted potency alone
selected a top-100 with median net charge +12.0 against +4.1 for measured-potent
AMPs — outside the oracle's training support and in the region where peptides are
characteristically hemolytic. 441 candidates were rejected by this filter in the
shipped run.

**Internal diversity (top-100).** No two selected candidates exceed 0.80
similarity to each other. Since 25 of the 100 are drawn at random for assay, a
cluster of near-identical peptides would convert one design idea into a
correlated bet; mutual distance makes the drawn subset informative about the
model rather than about a single motif.

## 4. Manual intervention

**None at the sequence level.** No peptide was hand-picked, hand-edited,
reordered, or removed from either the library or the top-100 list. Both files are
produced end-to-end by `uv run generate` with a fixed default seed.

Human judgement was exercised only in method design: model size and architecture,
the sampling temperature schedule, the three ranking terms and their per-category
weights, the similarity margin, the envelope percentile cutoffs, and the data
filters described above. None of these were tuned against a leaderboard or a
held-out competition metric — the aggregation score is withheld until Phase 1
closes, so there was nothing to tune against. They were set by reasoning from the
published scoring criteria and from the measured data.

## 5. AI assistance disclosure

This submission was developed with the assistance of Claude (Anthropic), used as
a coding and analysis assistant for implementation, data inspection, and
validation. All method decisions, filters, and the interpretation of results were
reviewed and directed by the human author. The generated peptides themselves are
the output of the trained model described above, not of a language assistant.
