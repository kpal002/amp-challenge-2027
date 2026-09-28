# Challenger-guided design: retrospective experiment

This experiment tests whether a learned warning about prediction error improves
candidate selection beyond the current potency-model recipe and a bootstrap
ensemble. It does not change the submission generator, checkpoints, or outputs.
Generator guidance is a subsequent experiment, conditional on supporting evidence.

## Colab workflow

Open `notebooks/challenger_colab.ipynb` in Colab and choose a GPU runtime. The
notebook accepts a ZIP of this checkout, so unpublished local code can be used
without pushing anything. Create that ZIP locally:

```bash
python scripts/bundle_challenger.py
```

Upload `colab_bundle/challenger.zip` when prompted. The notebook mounts Drive and
stores embeddings, fitted models, manifests, and reports under a directory you
choose there. Resume by running the same cells with the same ZIP and work
directory. Source edits or parameter changes require a fresh experiment directory;
do not discard an evaluated test result and call a later split untouched.

Local equivalent (ESM extraction uses CUDA when available, CPU otherwise):

```bash
uv sync --group research
uv run --no-sync python -m amp_challenge_2027.research prepare
uv run --no-sync python -m amp_challenge_2027.research embed
uv run --no-sync python -m amp_challenge_2027.research develop \
  --embeddings research_runs/challenger/embeddings/vectors.npz
```

For a quick descriptor-only comparison, omit `--embeddings`; its results are
stored separately. The default species is `E. coli`. Repeat `develop` with
`--species 'S. aureus'` or `--species 'P. aeruginosa'` for prespecified secondary
tasks. Commands fail if a task has too few strict labels for a useful comparison.
No model is downloaded until `embed`; ESM defaults to the 35M checkpoint pinned
to commit `6fbf070e65b0b7291e7bbcd451118c216cff79d8`. A different model requires
its full commit SHA and a new work directory. Partial extraction resumes from
atomic per-batch files; keep model, software, device, and batch size unchanged.

## Frozen experimental protocol

- Primary task: E. coli; primary endpoint: measured MIC <= 16 uM among 50 selected
  held-out candidates. Secondary tasks: S. aureus and P. aeruginosa. These are
  species-level historical labels, not the exact competition panel.
- Split all 4,121 eligible sequences into connected components at Levenshtein
  ratio >= 0.70. Every qualifying pair, including transitive bridges, remains
  in one partition. The threshold is a similarity definition, not biological
  proof of family membership. Giant components are reported, never broken up.
- Assign components by sequence counts only to approximately 55% training,
  15% calibration, 15% validation, 15% final test; seed 42. No activity labels are
  used to balance the split.
- Use confirmed modification reporting for primary labels. Unknown modification
  status remains a separate sensitivity scenario. GRAMPA's `value` in the bundled
  file is already log10(uM), despite `unit=uM`; do not transform it again.
- Within each species, remove duplicate sequence/strain/value observations,
  take the median within each strain, then the median across observed strains.
  This cannot identify every duplicate publication or harmonize missing assay
  conditions. Missing strain metadata stays one unspecified-strain group.
- Fit descriptor ridge and MLP models on training families only. Each scaler is
  fitted inside its own training pipeline. The MLP uses the existing 128/64
  architecture; its internal early-stopping split is within training data only.
- Fit five family-bootstrap ridge/MLP blends as the ordinary ensemble. Train
  contrasting ridge predictors with unknown modification records included,
  individual sources omitted, and ESM embeddings when supplied. Sources can
  share underlying experiments, so this is a sensitivity test, not lab transfer.
- Retain a contrasting predictor only if its calibration RMSE is at most 1.5x
  the ensemble's calibration RMSE. Report all exclusions.
- Train a nonnegative ridge error auditor on calibration families, targeting
  `max(0, measured_log_MIC - ensemble_prediction)`. Features are bootstrap spread,
  model disagreement, source/modification sensitivity, and distance to all
  training sequences available to the species models. No generated sequence
  receives an invented activity label.
- Compare four scores (lower is preferred): refitted potency blend, ensemble
  mean, ensemble mean + lambda * bootstrap SD, and ensemble mean + lambda *
  auditor warning. Tune both penalty coefficients on validation, using the
  same grid {0, 0.5, 1, 2}; ties prefer lower selected median MIC, then smaller
  coefficient. Zero allows the experiment to reject a useless warning.
- With embeddings, add ESM alone, an equal-weight mixture of the descriptor
  ensemble and ESM, and that mixture plus a tuned disagreement penalty. Its
  spread is the standard deviation of a model mixture assigning half the weight
  to ESM and half across descriptor ensemble members. These controls have the
  same representation access as the challenger. The representation-matched
  conservative ensemble is the primary comparator whenever ESM is enabled.
- Freeze coefficients, credible-model choices, and model hashes before test.
  Final evaluation does not refit or tune any model. Running development never
  aggregates or reports test activity labels.

The refitted potency blend is the current *potency-model recipe*, not the full
submission pipeline. Loading its shipped ranker would leak evaluation labels.
This first experiment does not recreate AMP-likeness ranking, the envelope, or
stratified selection. It isolates the incremental value of the error auditor.

## Reading results

Each task writes `DEVELOPMENT.md`, a detailed JSON report, candidate predictions,
fit provenance, and `frozen_policy.json` under `results/{descriptors,esm}/<species>`.
Development comparisons are tuned and exploratory. Do not report them as final
generalization performance.

Report activity yield, median measured MIC, distance to training, fixed
novelty-stratum comparisons, and the auditor's ability to flag fourfold potency
overprediction. The latter is assessed both across the whole evaluation pool
and among high-scoring candidates. A family-bootstrap interval compares the
challenger with the representation-matched conservative ensemble when ESM is
enabled (descriptor ensemble otherwise); it conditions on fitted models and
omits training variability. It is not a formal confidence guarantee.

Random-25 simulation is included as a secondary diagnostic. For fixed known
outcomes, its expected mean equals the top-list mean. It does not model shared
future assay failures or prove that diversity improves average potency.

Inspect development results before unlocking test. A worthwhile signal is an
improvement over the potency recipe, ESM alone, and the representation-matched
conservative ensemble,
without relying solely on reduced novelty, alongside useful error discrimination.
If the chosen challenger penalty is zero, the auditor added no selection value
under this tuning protocol. Treat inconclusive or negative results as such.

After settling the approach, explicitly evaluate the frozen policy once:

```bash
uv run --no-sync python -m amp_challenge_2027.research evaluate-test \
  --embeddings research_runs/challenger/embeddings/vectors.npz
```

Use exactly the same options as development. Cached final results are reused.
Changing code or hyperparameters after seeing test outcomes makes that test
development data, even if a new output directory lets the code run again.

ESM may have encountered held-out sequences during unsupervised pretraining.
No HC50 claim follows from these experiments. Retrospective success would justify
testing generator guidance; it would not establish activity of generated peptides.

## Sources

- [GRAMPA data documentation](https://github.com/zswitten/Antimicrobial-Peptides)
  defines modification-reporting uncertainty and source/strain fields.
- [Meta ESM-2 model card](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
  describes the pretrained checkpoint; its MIT license and upstream pretraining
  data must be disclosed if this model becomes part of the submission.
- [Conformal prediction under feedback covariate shift](https://doi.org/10.1073/pnas.2204569119)
  motivates care with model-guided selection. This implementation does not
  implement its coverage guarantees.
