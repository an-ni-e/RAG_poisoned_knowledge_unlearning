# RAG Poisoned Knowledge: Analysis, Tests and Figures

Notebook: `RAG_Poisoned_Knowledge_Analysis.ipynb`

Companion to `RAG_Poisoned_Knowledge_Unlearning.ipynb` (the main experiment). This notebook re-analyses the results the main notebook already produced. It does **not** call the language model, so it runs in seconds, with or without a GPU.

## Further Analysis, Tests and Figures

It adds what the main notebook lacked:

1. one shared set of helper functions (no duplicate definitions)
2. unit tests and data-integrity checks, so every number is verified before it is reported
3. an auditable label-correction table (`label_auto` is never overwritten)
4. metrics whose denominators are computed from the data, not typed in
5. a robustness check on the answer classifier
6. three figures: the main results with uncertainty, a per-fact map of what the poison and the unlearning do, and the threshold plot

## How to run

Use the same Colab session as the original notebook (the `data/` and `results/` folders must exist), or upload the outputs zip from the main notebook, unzip it, and run all cells top to bottom.

Note: the first code cell runs `!unzip rag_unlearning_outputs.zip`. 

**Required inputs**

```
data/documents.csv
data/queries.csv
results/uk_similarities.csv
results/unlearning_setup.json
results/{clean, poisoned, removal, unlearn_refusal, unlearn_quarantine}.csv
```

Optional (exploratory conditions, used if present):
`results/explore_constraint_only_refusal.csv`, `results/explore_poisoned_reversed.csv`, `results/explore_quarantine_reversed.csv`

Requires only `numpy`, `pandas` and `matplotlib` (preinstalled on Colab). The seed is 42.

## What the notebook does

| # | Section | Purpose |
|---|---|---|
| 1 | Load the saved results | Loads queries, documents, trigger threshold (tau = 0.724) and the 8 result files. Stops with a clear error if a file is missing. |
| 2 | Shared helpers and the label-correction table | Wilson intervals, classifier, fact-level bootstrap, metric builder, pairwise comparison, and the table of manual label corrections. |
| 3 | Unit tests | 29 hand-worked assertions (classifier edge cases, Wilson intervals, correction logic, toy metric frames). The notebook stops if one fails. |
| 4 | Integrity checks on the saved results | 34 PASS/FAIL checks (see below). |
| 5 | Apply the label corrections | Produces `label_auto` (classifier output, never changed) and `label_final` (used by all tables and figures). |
| 6 | Metrics | Main results table, interventions compared against their own baseline, and sensitivity to the label version. |
| 7 | Classifier robustness check | Recomputes every label with document-reference digits ignored. |
| 8a | Uncertainty | Query-level Wilson intervals versus fact-level bootstrap intervals. |
| 8b | Document-order check | Paired analysis of the poison claim when the poisoned document is listed first versus last. |
| 9 | Figure 1 | Main results with 95% Wilson whiskers. |
| 10 | Figure 2 | Per-fact map of outcomes for the poisoned KB, U-refusal and U-quarantine. |
| 11 | Figure 3 | Similarity to the forgotten text and the trigger threshold. |
| 12 | Save everything | Zips `data/`, `results/` and `figures/` into `rag_unlearning_outputs_v2.zip` (33 files). |

## Verification

**Unit tests (29 passed).** Examples: `4` must not match inside `14`; empty poison pattern for controls; `NaN` answers become `OTHER`; Wilson interval values and clamping; corrections only change `CONFLICT` rows and only to `CORRECT`; toy frames check every metric formula, including the denominators.

**Integrity checks (34/34 passed).** Highlights:

- `queries.csv` has 50 unique IDs, and every condition has the same 50 queries in the same order
- every stored label equals the label recomputed from the answer text
- no poisoned document is retrieved in `clean` or `removal`, and `removal` reproduces `clean` exactly
- a poisoned document is retrieved for every targeted query in `poisoned`
- untriggered rows in both unlearning conditions are identical to the poisoned condition
- trigger flags match `similarity >= tau`
- `tau` equals the pre-registered rule (midpoint of the lowest targeted-direct and highest control-direct similarity)

**Classifier robustness.** 0 of 400 labels change when document-reference digits (e.g. "document 2", "[2]") are ignored.

## Label corrections

`results/label_corrections.csv` lists every manual change, with a reason. Only `CONFLICT` answers can be corrected, and only to `CORRECT`.

| Tier | Count | Meaning |
|---|---|---|
| strict | 5 | Answer gives the right value and cites the poisoned value only to reject it (F07 paraphrase 1; F09 paraphrase 1 in four conditions) |
| borderline | 2 | Right value picked but a discrepancy is flagged (F06 paraphrase 2, F10 direct, reversed-order run) |

`LABEL_VERSION` (default `"strict"`) selects which tiers are applied: `auto`, `strict`, or `strict+borderline`.

## Main results (strict labels)

| | clean | poisoned | removal | U-refusal | U-quarantine |
|---|---|---|---|---|---|
| Poisoned docs retrieved (of 30) | 0 | 30 | 0 | 30 | 30 |
| ASR, poison claim only (of 30) | 0 | 6 | 0 | 16 | 4 |
| ASR-any, incl. conflict (of 30) | 0 | 10 | 0 | 17 | 5 |
| Targeted CORRECT (of 30) | 30 | 20 | 30 | 2 | 25 |
| Affected: no longer show poison (of 10) | 10 | 0 | 10 | 0 | 5 |
| Regressed, previously-correct (of 20) | 0 | 0 | 0 | 18 | 0 |
| Controls CORRECT (of 20) | 20 | 20 | 20 | 19 | 20 |

The "affected set" is the 10 targeted queries (6 facts) whose answer showed the poison in the poisoned condition. Unlearning triggered on 9 of the 10 in both unlearning conditions.

### Sensitivity to the label version

| | auto | strict | strict+borderline |
|---|---|---|---|
| Poisoned baseline ASR-any (of 30) | 12 | 10 | 10 |
| Affected set (queries / facts) | 12 / 8 | 10 / 6 | 10 / 6 |
| U-quarantine: no longer show poison | 6 of 12 | 5 of 10 | 5 of 10 |
| U-refusal: no longer show poison | 1 of 12 | 0 of 10 | 0 of 10 |
| U-refusal: regressed | 17/18 | 18/20 | 18/20 |
| U-quarantine: regressed | 0/18 | 0/20 | 0/20 |

The qualitative conclusions do not depend on the label version.

### Exploratory comparisons (against their own baseline)

| | Constraint-only refusal | Quarantine, reversed order |
|---|---|---|
| ASR-any (of 30) | 1 | 2 |
| Targeted CORRECT (of 30) | 3 | 28 |
| Regressed previously-correct | 17/20 | 0/18 |

### Uncertainty

`results/uncertainty_table.csv` reports, for each condition and measure, the Wilson interval over queries and a percentile bootstrap that resamples whole **facts** (queries about one fact are not independent). Example: poisoned ASR-any is 10/30 = 33% (Wilson 19-51%, fact bootstrap 17-50%). With only 10 facts and proportions near 0 or 1 the bootstrap can collapse to a point, so the Wilson interval is the safer one to report.

### Document-order check

For the same 30 targeted queries, the poison claim appeared in 18 answers when the poisoned document was listed last and in 4 when it was listed first (strict labels; exact McNemar p = 0.0005). The reversal is a deterministic swap, not a randomised design, and it is confounded with similarity rank, so treat these numbers as descriptive and do not compare them directly with the fixed-order tables.

## Figures

| File | Content |
|---|---|
| `figures/fig1_main_results.png` | Poison claim in answer, targeted answered correctly, controls answered correctly, for the five main conditions, with 95% Wilson whiskers (colour-blind-safe palette). |
| `figures/fig2_per_fact_map.png` | 10 facts x 3 phrasings for the poisoned KB, U-refusal and U-quarantine. Cell letters: C correct, P poison only, B both claims, R refusal/other. A dot marks queries where unlearning did not trigger. |
| `figures/fig3_similarity_threshold.png` | Every query's similarity to the nearest forgotten text, with tau and the calibration bounds a and b in the legend. Red outlines mark queries on the wrong side of the threshold. |

`figures/main_results.png` is the earlier figure from the main notebook (original labels) and is kept for reference.

## Output files

```
results/
  label_corrections.csv            every manual label change, with reason and tier
  summary_metrics_final.csv        main metrics table (label_final)
  label_sensitivity.csv            auto vs strict vs strict+borderline
  classifier_robustness_changes.csv  labels that change under the normalised classifier (empty)
  uncertainty_table.csv            Wilson and fact-level bootstrap intervals
  order_paired_analysis.csv        paired first-vs-last analysis
figures/
  fig1_main_results.png, fig2_per_fact_map.png, fig3_similarity_threshold.png
rag_unlearning_outputs_v2.zip      data/ + results/ + figures/ (33 files)
```

## Caveats

- Ten facts and 50 queries from one generator model and one retriever: intervals are wide, and repeated phrasings of the same fact are not independent. Treat the results as a case study.
- Labels are regex-based with a manual correction table; only `CONFLICT` rows were reviewed for correction.
- The trigger threshold was calibrated on the "direct" queries only; the paraphrase columns in Figure 3 are the held-out check. Some targeted paraphrase queries fall below tau (not triggered), and one same-domain control paraphrase falls above it (triggered).
- The order check and the exploratory conditions are descriptive, post hoc analyses.
