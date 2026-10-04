# Evaluation

```bash
python -m app.evaluation.cli run --synthetic --summary
python -m app.evaluation.cli run --cases my_cases.jsonl --k 3 --out report.json
```

## Case format (JSONL or CSV)

| Field | Required | Meaning |
|---|---|---|
| `clinical_note` | yes | Note text |
| `expected_code` | no | Expected code. Empty means "nothing should be coded" (negative case). |
| `expected_dataset`, `expected_version` | yes | Dataset to evaluate against |
| `notes` | no | Free text |
| `expected_concept`, `expected_status` | no | Enables concept-extraction metrics |
| `coder_code` | no | A human coder's code, used for agreement |

## Metrics, per stage

| Stage | Metrics |
|---|---|
| Concept extraction | `concept_recall` (expected concept found, word overlap ≥ 0.6), `status_accuracy` |
| Retrieval | `recall@k`, `recall@10`, `mrr` of the expected code among hybrid candidates |
| Reranking | `top1_accuracy`, `recall@k`, `mrr` after rule/specificity filtering and rerank |
| Final selection | `top1_accuracy`, `top3_recall` (suggestion + alternatives), `mrr`, `negative_case_accuracy`, `hierarchy_accuracy` (prediction is the expected code, an ancestor or a descendant) |
| Safety | `unsupported_code_rate` (every returned code re-checked in the DB), `rule_violation_rate` |
| Agreement | `human_coder_agreement` (top-1 = `coder_code`) |

The per-case `details` show the retrieval, rerank and final ranks, so you can tell which stage
failed.

## Synthetic suite (`app/synthetic/eval_cases.py`, 18 cases)

Measured on 2026-10-03 against the synthetic 2024 dataset (hashing embeddings):

| Metric | Value |
|---|---|
| retrieval recall@3 / recall@10 / MRR | 0.94 / 1.00 / 0.88 |
| reranking top-1 / MRR | 1.00 / 1.00 |
| final top-1 / top-3 / negative-case accuracy / hierarchy accuracy | 1.00 / 1.00 / 1.00 / 1.00 |
| unsupported-code rate / rule-violation rate | 0.00 / 0.00 |
| concept recall / status accuracy | 1.00 / 1.00 |
| human-coder agreement (cases with `coder_code`) | 1.00 |

The difference between retrieval and reranking is the point of the stage split. For example,
"Lobar consolidation on chest imaging" ranks A01.9 5th in retrieval. The specificity guard and
reranker move it to 1st, because no side is documented.

> **These cases validate system behaviour only.** The synthetic classification was written for
> this project, so its results say nothing about real ICD-10-CA coding accuracy. Real accuracy
> must be measured with a licensed dataset and cases coded by qualified coders.
