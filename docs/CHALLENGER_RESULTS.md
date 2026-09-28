# Challenger experiment: development snapshot

Run on 2026-09-27. **Exploratory validation results, not final test or generated-peptide efficacy.**

All methods select 50 historical candidates. A hit means the species-level measured MIC label is <= 16 uM. Model families, source hashes, fitted-model hashes, and runtime versions are recorded in each JSON report.

| Method | E. coli | S. aureus | P. aeruginosa |
|---|---:|---:|---:|
| Refitted potency recipe | 66% | 56% | 84% |
| Descriptor conservative ensemble | 82% | 68% | 74% |
| ESM alone | 80% | 52% | 84% |
| Descriptor + ESM mean | 84% | 66% | 80% |
| Descriptor + ESM conservative ensemble | 86% | 66% | 82% |
| Learned challenger + ESM | 86% | 66% | 86% |

The challenger ties the representation-matched conservative ensemble on the primary E. coli endpoint. It improves P. aeruginosa by two hits (43/50 versus 41/50), but improves the original potency recipe there by only one hit. S. aureus is worse than the descriptor-only conservative ensemble. These results do not establish an incremental benefit sufficient to justify generator fine-tuning yet.

The representation-matched controls were added after inspecting the initial development run, to check whether ESM access alone explained the apparent benefit. The initial artifacts and their source bundle are retained under `research_runs/challenger_initial/`. No real-data final test was evaluated in either run.

## Uncertainty and error detection

| Species | Challenger minus ESM conservative ensemble, exploratory 95% family-bootstrap interval | Auditor error AUC in high-scoring pool |
|---|---:|---:|
| E. coli | [-8%, +8%] | 0.605 |
| S. aureus | [-16%, +20%] | 0.414 |
| P. aeruginosa | [-6%, +14%] | 0.697 |

All intervals include zero. They condition on fitted models, omit training variation, and do not correct for policy tuning on validation. The warning event is a fourfold overprediction of potency. Chance discrimination has AUC 0.5; S. aureus does not show a useful warning signal in its high-scoring pool.

## Reproduce or continue

- [Protocol and commands](CHALLENGER_EXPERIMENT.md)
- [Colab notebook](../notebooks/challenger_colab.ipynb)
- Build the upload ZIP with `python scripts/bundle_challenger.py`.
- Current local artifacts: `research_runs/challenger/results/esm/`.
- **Final test remains unopened.** Keep it that way while deciding whether to refine or reject the auditor.

Verified locally: 51 tests pass; cached ESM-2 35M extraction and all three species development runs complete; completed-run and embedding resume paths work. Colab cells compile and the upload ZIP validates. Actual Colab/GPU execution has not been exercised from this machine.

## Run identities

- E. coli: `8c5c055e4707427975cd8ad7708bced5ea6d59e094ba740941694fa42f5f027e`
- S. aureus: `12337c32e6d01986310fc620c1d32f2b791fb6e654c0548961ea799b364d9c64`
- P. aeruginosa: `d6d14128ed93021573b633a45c5058e1dd620f1e3139adb798f544f04b773315`
