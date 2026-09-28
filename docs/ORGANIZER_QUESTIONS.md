# Discussion post — ready to paste

Post on the competition Discussion page
(<https://www.kaggle.com/competitions/amp-challenge/discussion>). The first
section is a finding that affects the whole benchmark, not just us; the rest are
questions we need answered to finalise our submission.

Trim or reword freely — it is written to be useful to the organizers rather than
to advertise us.

---

**Subject: Disulfide ambiguity in submitted top-100 lists, and three questions on
data return and the identity rule**

Hi — thanks for organising this. One observation we think is worth raising before
synthesis begins, then three questions.

### 1. Multi-cysteine candidates may produce uninterpretable assay data

The peptide constraints require linear, unmodified peptides with free termini and
no non-canonical residues, and there is no controlled oxidation or refolding step
in the pipeline. Under those conditions a submitted sequence containing two or
more cysteines is not a single molecule: it is an undefined mixture of disulfide
isomers (plus intermolecular species), and one with an odd cysteine count carries
a free thiol that can oxidise in the plate. The resulting MIC and HC50 values are
hard to attribute to a defined structure whichever way they come out.

We think this is likely to affect more than one team, for a structural reason.
Cysteine-rich AMPs — defensins and related families — are among the most potent
entries in the public MIC compilations, but they are potent **as folded,
disulfide-bonded molecules**. Any potency model fitted on those data learns
cysteine as a positive signal and will preferentially rank cysteine-rich
candidates, even though the submission rules forbid the structure that makes them
potent. In our own pipeline this was quantitative: our generated library is 26.3%
cysteine-containing and 12.6% multi-cysteine, closely matching the reference
database (26.4% / 13.3%), but our *selected* top-100 came out 46% and 40%. The
selection stage, not the generator, did that. Roughly 10 of our 25 randomly drawn
peptides would have been undefined mixtures.

We now exclude cysteine entirely from our top-100. It cost 0.037 log10 units of
predicted potency, which is 5.6% of our potency model's own cross-validated RMSE
— i.e. nothing measurable — so for us this was close to free.

Two suggestions, offered rather than urged:

- It may be worth reporting cysteine counts in the compliance screening summary,
  so teams can see this before the deadline rather than after synthesis.
- If multi-cysteine peptides are synthesised, recording the oxidation state or
  handling conditions in the released dataset would make those rows much more
  reusable, given the data is going out under CC-BY 4.0 and into DBAASP/APD3.

### 2. Is per-peptide, per-strain data returned to teams?

The award categories are computed on team means, and the overview says advancing
teams "receive full experimental results for all tested peptides". Could you
confirm what that resolves to concretely:

- Do teams receive **per-peptide, per-strain MIC values** and **per-peptide
  HC50**, or only the five aggregate category scores?
- Are the identities of the 25 randomly selected peptides disclosed to the team?
- Roughly what timeline should we expect (the schedule shows assays Mar–Apr 2027)?

This matters to us because we are considering submitting a top-100 deliberately
structured as a balanced comparison between two candidate-selection policies, so
that your uniform random draw of 25 yields a randomised, single-lab, blinded test
of which selection rule actually produces active peptides — with the generator
held fixed. That speaks directly to the stated aim of identifying "which
computational metrics, ranking strategies, and filtering rules are actually
associated with experimental antimicrobial efficacy", and to the reproducibility
problem in the competition's framing.

That design only yields anything if sequence-level results come back. If they do
not, we will ship a conventionally optimised list instead.

### 3. How does the identity-rule replacement work in practice?

The rules say candidates exceeding 80% identity to the reference database "are
treated as invalid and replaced by the next valid candidate". If a submitted file
contains exactly 100 entries, what does "the next valid candidate" resolve to —
is a replacement drawn from the submitted 50,000-sequence library, or is the
tested cohort reduced below 25 for that team?

We ask because we enforce a 0.75 ceiling ourselves, partly as a margin against
the difference between `Levenshtein.ratio` (as implemented in the template's
`verify_submission.py`) and MMseqs2 alignment identity (as specified in the
competition proposal), which are not the same measure. If replacement does draw
from the library, it would also perturb any team's balanced design, so it would
be useful to know.

### 4. Offer

If the two-arm design above is of interest, we are happy to (a) pre-register the
arm assignment publicly before the deadline, and (b) share the assignment so it
can be pooled with any other team willing to do the same. Pooled across even a
few teams, this would give the benchmark a properly powered answer on selection
policy that no single team can produce from 25 peptides.

Our submission and full methodology are public at
<https://github.com/kpal002/amp-challenge-2027> (MIT).

Thanks,
[name, affiliation]
