# Submission checklist

Deadline: **2026-09-30, 15:00 PDT**. Everything on the engineering side is done
and verified; the items below need account access and so cannot be automated.

## 1. Register (blocker — do this first)

The competition rejects gmail / hotmail / yahoo for primary registration. An
**institutional email address** is required.

Nothing else on this list matters until this is resolved, because an unregistered
team cannot submit regardless of code quality.

Portal: <https://www.kaggle.com/competitions/amp-challenge>

## 2. Confirm how many categories are required

The main template exposes a single `generate` entry point; the newer starter kits
expect five category-suffixed entry points. We ship **all five**, so either
answer works — but check the Kaggle page so you know what to attach.

| Entry point | Output directory |
|---|---|
| `generate` (alias of `generate_broad_spectrum`) | `generate/` |
| `generate_broad_spectrum` | `generate_broad_spectrum/` |
| `generate_gram_pos` | `generate_gram_pos/` |
| `generate_gram_neg` | `generate_gram_neg/` |
| `generate_mdr` | `generate_mdr/` |
| `generate_therapeutic` | `generate_therapeutic/` |

## 3. Push the repository

The work is committed locally on `main` as a single initial commit. Create an
empty GitHub repository, then:

```bash
cd ~/amp-challenge
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

Then in the repository's **Settings → Collaborators**, grant read access to
`@RasmusML` and `@szymczakpau`.

For co-authorship eligibility the repository must be **public** and carry a
permissive OSI licence — MIT is already in place as `LICENSE`.

## 4. Verify what the organizers will verify

Already run and passing locally, but re-run after pushing so the check reflects
the pushed remote rather than the local tree:

```bash
cd /path/to/amp-challenge-2027-template
uv run python scripts/verify_submission.py https://github.com/<your-user>/<your-repo>
```

Expected final line: `All checks passed. Submission is valid!`

## 5. Attach the written material

| Requirement | File |
|---|---|
| Abstract summarising the method | `docs/ABSTRACT.md` |
| Top-100 selection and ranking procedure | `docs/DATA_AND_FILTERS.md` §3, plus README "Candidate selection" |
| Training data, external databases, filters, manual intervention | `docs/DATA_AND_FILTERS.md` |
| AI-assistance disclosure | `docs/DATA_AND_FILTERS.md` §5 |

## 6. Full-requirements audit

| Requirement | Status |
|---|---|
| Public GitHub repo following the template | pending push (step 3) |
| Permissive OSI licence specified | done — MIT |
| Trained model weights in repo | done — `checkpoint/generator.npz`, `checkpoint/ranker.npz` |
| Inference code and usage documentation | done — `README.md` |
| Fixed default seed, identical output on re-run | done — verified byte-identical |
| `uv sync` + entry point reproduces the library | done — verified with the official script |
| Full training-data disclosure | done — all public, both source files committed |
| Non-public data released | n/a — no non-public data used |
| Read access for `@RasmusML`, `@szymczakpau` | pending push (step 3) |

## 7. Peptide constraints audit

All verified programmatically across all five categories.

| Constraint | Status |
|---|---|
| 20 standard amino acids only | enforced in the sampler; validated |
| Length 8–50 | structural — EOS masked below 8, forced at 50 |
| Linear peptides only | no modifications emitted; sequence-only output |
| No terminal modifications | free termini by construction |
| No non-canonical residues or conjugates | alphabet restricted to 20 residues |
| Unique sequences only | 50,000/50,000 unique in every category |
| Top-100 ≤ 80% identity to reference | max 0.750 in every category (0.05 margin) |
