# Abstract

**A category-conditioned peptide language model with distribution-constrained
candidate selection**

We generate antimicrobial peptide libraries with a 1.8M-parameter
decoder-only transformer trained on 39,655 public AMPs, conditioned on a control
token that selects the competition category. Peptides are 8–50 residues over a
20-letter alphabet, so the model is deliberately small: at this data scale larger
models mostly memorise the reference database. Validity is enforced inside the
sampler rather than by rejection — the end-of-sequence token is masked until a
draw reaches eight residues and forced at fifty — so every sample is compliant by
construction.

Candidate ranking combines three rank-normalised terms: predicted potency from a
log₁₀ MIC regressor, an AMP-likeness classifier trained against
composition-matched shuffles (out-of-fold AUC 0.785 on that order-only task), and
a hemolysis-informed selectivity proxy. The potency model achieves out-of-fold
Spearman ρ 0.557 on a random split but only **ρ 0.468, RMSE 0.656 log₁₀ units**
when peptides are grouped by sequence similarity and split by group — roughly a
factor of 4.5 in µM. We quote the clustered figure as the oracle's real accuracy,
since AMP databases are dense with homologues and a random split leaves
near-identical peptides in the training fold. Because the competition's aggregation score is withheld until
Phase 1 closes, we did not tune against a ranking function; we instead optimise
for robustness across the published metric families independently.

The submission's distinguishing choice is a **plausibility envelope**. Ranking on
predicted potency alone drove selection into the upper tail of the
cationic/amphipathic distribution — median net charge +12.0 against +4.1 for
peptides with measured MIC ≤ 10 µM — where the potency oracle extrapolates beyond
its training support and where peptides are characteristically hemolytic. We
therefore restrict top-100 candidates to the central 95% range of
measured-potent AMPs across ten physicochemical descriptors. This lowers nominal
predicted potency while keeping candidates in the region where the oracle has
evidence and where embedding-based comparisons to known potent peptides are
meaningful.

Bounding *where* candidates sit proved insufficient on its own. Evaluated with
`seqme` — the organizers' own framework, independent of our scorers —
score-greedy selection inside the envelope still yielded a top-100 with precision
0.97 and **recall 0.17**: sequences that differ from one another while sharing
nearly identical charge and length, and which can therefore fail for the same
reason. Since 25 of the 100 are drawn at random for assay, that homogeneity turns
the draw into a correlated bet. We therefore stratify selection over 64
physicochemical strata, taking the best admissible candidate from each in turn
from among the top 12,000 by score. This takes the top-100 from FBD 10.87 and
recall 0.17 under score-greedy selection to **FBD 2.02, MMD 2.75, recall 0.85,
diversity 0.844 and conformity 0.592** — better than the measured-potent reference
cohort on FBD and MMD (2.94 / 10.89) and comparable on the rest — at a cost of
0.25 log₁₀ units in top-10 predicted potency, which is inside the oracle's error.

Two data decisions follow the same logic. The potency model is fitted only on
*unmodified* GRAMPA measurements, because 43% of that corpus is C-terminally
amidated while the competition requires free termini; training on amidated
measurements would model a molecule we cannot submit. And the MDR category is
defined from measurements rather than labels — peptides assayed against ≥3
distinct species with median MIC ≤ 10 µM — because the reference database's
Gram-positive and Gram-negative activity labels are mutually exclusive and so
cannot express cross-spectrum activity.

The category names match the competition's award categories but the labels behind
them are proxies and claim less than the names imply. `mdr` encodes cross-species
**breadth**, not activity against drug-resistant isolates: no resistant strain
appears in the training data. `therapeutic` is primarily cysteine-free reference
peptides, and no measured HC50 enters the pipeline at any point, so it reflects a
structural prior and a hand-specified selectivity proxy rather than demonstrated
low hemolysis.

The generated library matches the reference AMP distribution on net charge (2.90
vs 2.99), hydrophobic moment (0.504 vs 0.560) and length (18 vs 15), while
separating cleanly from decoys under the AMP-likeness model. Independently, under
seqme against a held-out half of the reference database it reaches FBD 0.432 and
MMD 0.300, versus 0.227 / 0.126 for real-AMPs-against-real-AMPs and 1.416 / 3.79
for composition-matched shuffles. All 50,000 sequences are unique, mean
pairwise similarity is 0.250, and no top-100 candidate exceeds 0.750 similarity to
any reference AMP — below the required 0.80, since the template implements the
rule as Levenshtein ratio while the competition proposal specifies MMseqs2
alignment identity.

Inference is pure NumPy against committed weights; PyTorch appears only in
training. This keeps `uv sync` light and platform-neutral and makes generation
deterministic by construction, which we verify locally by running the entry point
twice and byte-comparing both FASTA files. A parity test confirms the NumPy
forward pass reproduces the trained PyTorch logits to 8×10⁻⁶ with identical
argmax. No sequence was hand-picked or hand-edited at any stage.
