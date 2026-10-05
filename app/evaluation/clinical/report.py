"""Aggregate per-scenario results into the clinical evaluation report + release safety gates.

Quality metrics are reported as measured (never hidden or averaged into a single score).
Safety gates are hard: any non-zero value fails the release, whatever the quality metrics say.
"""

from collections import Counter, defaultdict
from typing import Any

from app.evaluation.clinical.evaluator import MODES, STAGE_ORDER
from app.evaluation.clinical.oracles import LatencySummary, rate

DISCLAIMER = (
    "All scenarios and classification data are ORIGINAL SYNTHETIC test fixtures. The metrics "
    "validate system behaviour and safety only. They are not real ICD-10-CA coding accuracy, "
    "not coder agreement and not clinical diagnostic accuracy. Latencies come from a small "
    "synthetic dataset in a local container and are not a production SLA."
)
RETRIEVAL_TAGS = (
    "direct",
    "paraphrase",
    "typo",
    "ambiguous",
    "specificity",
    "rules",
    "negation",
    "symptom_only",
    "multiple",
    "version",
    "adversarial",
    "code_query",
)
UNCERTAIN = {"uncertain", "suspected"}


def _ranks_summary(ranks: list[int | None]) -> dict[str, Any]:
    n = len(ranks)

    def recall(k: int) -> float | None:
        return rate(sum(1 for r in ranks if r is not None and r <= k), n)

    return {
        "queries": n,
        "recall@1": recall(1),
        "recall@3": recall(3),
        "recall@5": recall(5),
        "recall@10": recall(10),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4) if n else None,
    }


def _status_metrics(pairs: list[tuple[str, str]], statuses: set[str]) -> dict[str, Any]:
    expected = [p for p in pairs if p[0] in statuses]
    predicted = [p for p in pairs if p[1] in statuses]
    return {
        "expected": len(expected),
        "recall": rate(sum(1 for e, a in expected if a in statuses), len(expected)),
        "precision": rate(sum(1 for e, a in predicted if e in statuses), len(predicted)),
    }


def build_report(results: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    evaluated = [r for r in results if "final" in r]
    tags = Counter(t for r in results for t in r["tags"])

    # --- concepts -----------------------------------------------------------------------
    tp = sum(r["concepts"]["true_positives"] for r in evaluated)
    fp = sum(len(r["concepts"]["false_positives"]) for r in evaluated)
    fn = sum(len(r["concepts"]["false_negatives"]) for r in evaluated)
    pairs = [tuple(p) for r in evaluated for p in r["concepts"]["statuses"]]
    attributes: dict[str, list[bool]] = defaultdict(list)
    for r in evaluated:
        for item in r["concepts"]["attributes"]:
            attributes[item["attribute"]].append(item["ok"])
    concept = {
        "precision": rate(tp, tp + fp),
        "recall": rate(tp, tp + fn),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "assertion_accuracy": rate(sum(1 for e, a in pairs if e == a), len(pairs)),
        "assertion_pairs": len(pairs),
        "negation": _status_metrics(pairs, {"negated"}),
        "ruled_out": _status_metrics(pairs, {"ruled_out"}),
        "history": _status_metrics(pairs, {"history"}),
        "family_history": _status_metrics(pairs, {"family_history"}),
        "uncertainty": _status_metrics(pairs, UNCERTAIN),
        "documented": _status_metrics(pairs, {"documented"}),
        "attribute_accuracy": {
            name: {"checked": len(values), "accuracy": rate(sum(values), len(values))}
            for name, values in sorted(attributes.items())
        },
        "attribute_accuracy_overall": rate(
            sum(sum(v) for v in attributes.values()), sum(len(v) for v in attributes.values())
        ),
    }

    # --- retrieval (each expected code of each positive scenario is one query) ------------
    by_mode: dict[str, list[int | None]] = defaultdict(list)
    by_tag: dict[str, dict[str, list[int | None]]] = defaultdict(lambda: defaultdict(list))
    for r in evaluated:
        for code in r["retrieval_targets"]:
            ranks = r["retrieval"]["ranks"].get(code, {})
            for mode in MODES:
                by_mode[mode].append(ranks.get(mode))
                for tag in set(r["tags"]) & set(RETRIEVAL_TAGS):
                    by_tag[tag][mode].append(ranks.get(mode))
    retrieval = {
        "by_mode": {mode: _ranks_summary(by_mode[mode]) for mode in MODES},
        "by_tag": {
            tag: {mode: _ranks_summary(values[mode]) for mode in MODES}
            for tag, values in sorted(by_tag.items())
        },
        "semantic_status": dict(Counter(r.get("semantic_status") for r in evaluated)),
        "note": "exact retrieval only applies to code-like queries; vector is null when the "
        "dataset is not fully embedded in the configured provider's space",
    }

    # --- reranking -------------------------------------------------------------------------
    before, after = [], []
    improved = degraded = unchanged = removed = 0
    removed_cases = []
    for r in evaluated:
        for code in r["retrieval_targets"]:
            item = r["reranking"]["per_code"].get(code, {})
            b, a = item.get("retrieval_rank"), item.get("rerank_rank")
            if b is None:
                continue
            before.append(b)
            after.append(a)
            if a is None:
                if item.get("rejected_reasons"):
                    removed += 1
                    removed_cases.append(
                        {"id": r["id"], "code": code, "reasons": item["rejected_reasons"][:2]}
                    )
                degraded += 1
            elif a < b:
                improved += 1
            elif a > b:
                degraded += 1
            else:
                unchanged += 1
    promoted = []
    for r in evaluated:
        allowed = set(r["expected_codes"])
        if not allowed:
            continue
        for item in r["reranking"]["per_concept"]:
            if item["retrieval_top1"] in allowed and item["rerank_top1"] not in (allowed | {None}):
                promoted.append(
                    {"id": r["id"], "concept": item["concept"], "promoted": item["rerank_top1"]}
                )
    reranking = {
        "pairs": len(before),
        "top1_before": rate(sum(1 for b in before if b == 1), len(before)),
        "top1_after": rate(sum(1 for a in after if a == 1), len(after)),
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
        "correct_candidate_removed": removed,
        "correct_candidate_removed_cases": removed_cases,
        "incorrect_candidate_promoted": len(promoted),
        "incorrect_candidate_promoted_cases": promoted,
        "note": "a removed correct candidate is not necessarily an error: e.g. a more specific "
        "code rejected for undocumented detail; see failure analysis",
    }

    # --- final selection ---------------------------------------------------------------------
    positives = [r for r in evaluated if r["expected_codes"]]
    primaries = sum(len(r["final"]["primary_codes"]) for r in evaluated)
    false_positive_codes = sum(len(r["final"]["false_positives"]) for r in evaluated)
    expected_total = sum(len(r["expected_codes"]) for r in positives)
    hits = sum(len(r["final"]["hits"]) for r in positives)
    final = {
        "scenarios_passed": sum(1 for r in results if r.get("passed")),
        "scenarios": len(results),
        "pass_rate": rate(sum(1 for r in results if r.get("passed")), len(results)),
        "expected_code_recall": rate(hits, expected_total),
        "code_precision": rate(primaries - false_positive_codes, primaries),
        "exact_code_set_match": rate(
            sum(
                1
                for r in positives
                if not r["final"]["missed"] and not r["final"]["false_positives"]
            ),
            len(positives),
        ),
        "primary_codes_returned": primaries,
        "false_positive_codes": false_positive_codes,
        "must_not_return_violations": sum(
            len(r["final"]["must_not_return_violations"]) for r in evaluated
        ),
        "concept_association_errors": sum(len(r["final"]["association_errors"]) for r in evaluated),
        "duplicate_suggestions": sum(len(r["final"]["duplicates"]) for r in evaluated),
    }

    # --- rules -----------------------------------------------------------------------------
    by_rule: dict[str, list[bool]] = defaultdict(list)
    for r in evaluated:
        for rule in r["rules"]["expected_instructions"]:
            by_rule[rule].append(rule in r["rules"]["instructions_found"])
    expected_rejections = sum(len(r["rules"]["expected_rejected"]) for r in evaluated)
    fatal = [{"id": r["id"], **v} for r in evaluated for v in r["rules"]["fatal_violations"]]
    rule_targets = sum(r["database"]["counts"]["rule_targets"] for r in evaluated)
    invalid_targets = sum(r["database"]["counts"]["invalid_rule_targets"] for r in evaluated)
    rules = {
        "detection_accuracy": rate(
            sum(sum(v) for v in by_rule.values()), sum(len(v) for v in by_rule.values())
        ),
        "by_rule_type": {
            rule: {"expected": len(v), "detected": sum(v), "accuracy": rate(sum(v), len(v))}
            for rule, v in sorted(by_rule.items())
        },
        "candidate_rejection_accuracy": rate(
            sum(len(r["rules"]["rejected_ok"]) for r in evaluated), expected_rejections
        ),
        "candidate_rejections_observed": rate(
            sum(len(r["rules"]["rejected_observed"]) for r in evaluated), expected_rejections
        ),
        "fatal_rule_violations": len(fatal),
        "fatal_rule_violation_cases": fatal,
        "rule_violation_rate": rate(len(fatal), primaries),
        "rule_targets_checked": rule_targets,
        "invalid_target_codes": invalid_targets,
        "invalid_target_code_rate": rate(invalid_targets, rule_targets),
    }

    # --- specificity -------------------------------------------------------------------------
    unsupported = [{"id": r["id"], **u} for r in evaluated for u in r["specificity"]["unsupported"]]
    unsupported_codes = {(u["id"], u["code"]) for u in unsupported}
    fallback = [r for r in evaluated if "fallback" in r["tags"]]
    required = sum(len(r["missing_information"]["required"]) for r in evaluated)
    reported = sum(len(r["missing_information"]["reported"]) for r in evaluated)
    not_genuine = sum(len(r["missing_information"]["not_genuine"]) for r in evaluated)
    specificity = {
        "unsupported_specificity_rate": rate(len(unsupported_codes), primaries),
        "unsupported_specificity_by_attribute": dict(Counter(u["attribute"] for u in unsupported)),
        "unsupported_specificity_cases": unsupported,
        "safe_fallback_scenarios": len(fallback),
        "safe_fallback_rate": rate(
            sum(
                1
                for r in fallback
                if not r["final"]["missed"]
                and not r["final"]["false_positives"]
                and not r["specificity"]["unsupported"]
            ),
            len(fallback),
        ),
        "missing_information_recall": rate(
            sum(len(r["missing_information"]["found"]) for r in evaluated), required
        ),
        "missing_information_precision": rate(reported - not_genuine, reported),
        "missing_information_not_genuine": not_genuine,
    }

    # --- abstention -------------------------------------------------------------------------
    should = [r for r in evaluated if r["should_abstain"]]
    abstained = [r for r in evaluated if r["final"]["abstained"]]
    true_abstain = [r for r in should if r["final"]["abstained"]]
    # Abstaining is wrong only where a code was required; where the scenario lists only
    # acceptable alternatives, abstention is an acceptable outcome.
    wrongly = [r for r in abstained if r["expected_codes"]]
    abstention = {
        "abstention_scenarios": len(should),
        "precision": rate(len(abstained) - len(wrongly), len(abstained)),
        "recall": rate(len(true_abstain), len(should)),
        "system_abstained": len(abstained),
        "codes_returned_on_abstention_scenarios": sum(
            len(r["final"]["primary_codes"]) for r in should
        ),
        "false_positive_code_rate": rate(false_positive_codes, primaries),
        "wrongly_abstained": [r["id"] for r in wrongly],
        "acceptable_abstentions": [
            r["id"] for r in abstained if not r["should_abstain"] and not r["expected_codes"]
        ],
        "missed_abstentions": [r["id"] for r in should if not r["final"]["abstained"]],
    }

    # --- database safety ------------------------------------------------------------------
    db_counts: Counter[str] = Counter()
    for r in evaluated:
        db_counts.update(r["database"]["counts"])
    db_issues = [{"id": r["id"], **i} for r in evaluated for i in r["database"]["issues"]]
    database = {
        "returned_codes_checked": db_counts["returned"],
        "unsupported_codes": db_counts["unsupported"],
        "unsupported_code_rate": rate(db_counts["unsupported"], db_counts["returned"]) or 0.0,
        "cross_version_contamination": db_counts["cross_version"],
        "coding_system_contamination": db_counts["cross_system"],
        "non_ready_leakage": db_counts["non_ready"],
        "inactive_records": db_counts["inactive"],
        "title_mismatches": db_counts["title_mismatch"],
        "issues": db_issues,
        "dataset_resolution_errors": [
            {"id": r["id"], "error": r["error"]} for r in results if "error" in r
        ],
    }

    # --- status safety -------------------------------------------------------------------
    status_safety = Counter(v["issue"] for r in evaluated for v in r["status_safety"])

    # --- confidence -----------------------------------------------------------------------
    levels: dict[str, list[bool]] = defaultdict(list)
    signal_levels: dict[str, Counter[str]] = defaultdict(Counter)
    violations = []
    for r in evaluated:
        for item in r["confidence"]:
            levels[item["level"]].append(item["correct"])
            s = item["signals"]
            categories = {
                "vector_strong_lexical_weak": s["vector_strong"] and not s["lexical_strong"],
                "lexical_strong_vector_weak": s["lexical_strong"] and not s["vector_strong"],
                "all_signals_agree": s["agree"],
                "near_tie": s["near_tie"],
                "specificity_incomplete": s["specificity_incomplete"],
                "rule_conflict": s["rule_conflict"],
            }
            for name, on in categories.items():
                if on:
                    signal_levels[name][item["level"]] += 1
            if item["violation"]:
                violations.append({"id": r["id"], "code": item["code"], "level": item["level"]})
    confidence = {
        "levels": {
            level: {
                "count": len(levels.get(level, [])),
                "correct": sum(levels.get(level, [])),
                "precision": rate(sum(levels.get(level, [])), len(levels.get(level, []))),
            }
            for level in ("HIGH", "MEDIUM", "LOW")
        },
        "by_signal": {name: dict(counter) for name, counter in sorted(signal_levels.items())},
        "unjustified_high_or_above_max": len(violations),
        "violations": violations,
    }

    # --- evidence ---------------------------------------------------------------------------
    ev: Counter[str] = Counter()
    for r in evaluated:
        ev.update({k: v for k, v in r["evidence"].items() if isinstance(v, int)})
    evidence = {
        "items": ev["items"],
        "traceability": rate(ev["traceable"], ev["items"]),
        "incorrect_evidence": ev["incorrect"],
        "fabricated_evidence": ev["fabricated"],
        "provenance_accuracy": rate(ev["provenance_ok"], ev["provenance_total"]),
        "problems": [{"id": r["id"], **p} for r in evaluated for p in r["evidence"]["problems"]],
    }

    # --- failure analysis ---------------------------------------------------------------------
    failed = [r for r in results if not r.get("passed")]
    failures = {
        "failed_scenarios": len(failed),
        "by_primary_stage": {
            stage: count
            for stage, count in sorted(
                Counter(r["failure_stage"] for r in failed).items(),
                key=lambda item: STAGE_ORDER.index(item[0]),
            )
        },
        "by_any_stage": dict(Counter(s for r in failed for s in r.get("failure_stages", []))),
        "cases": [
            {
                "id": r["id"],
                "tags": r["tags"],
                "stage": r["failure_stage"],
                "details": [f"{stage}: {detail}" for stage, detail in r["failures"]],
                "expected": r["expected_codes"],
                "returned": r.get("final", {}).get("primary_codes"),
            }
            for r in failed
        ],
    }

    # --- latency -----------------------------------------------------------------------------
    latency: dict[str, LatencySummary] = defaultdict(LatencySummary)
    for r in evaluated:
        for stage, value in r["latency_ms"].items():
            latency[stage].values.append(value)

    report = {
        "meta": {**meta, "disclaimer": DISCLAIMER},
        "dataset": {
            "scenarios": len(results),
            "abstention_scenarios": sum(1 for r in results if r["should_abstain"]),
            "positive_scenarios": len(positives),
            "by_tag": dict(sorted(tags.items())),
            "by_dataset": dict(Counter(f"{r['coding_system']} {r['version']}" for r in results)),
        },
        "concept_extraction": concept,
        "retrieval": retrieval,
        "reranking": reranking,
        "final_selection": final,
        "rules": rules,
        "specificity": specificity,
        "abstention": abstention,
        "database_safety": database,
        "status_safety": dict(status_safety),
        "confidence": confidence,
        "evidence": evidence,
        "failures": failures,
        "latency_ms": {stage: summary.as_dict() for stage, summary in latency.items()},
    }
    report["safety_gates"] = safety_gates(report)
    report["safety_passed"] = all(g["passed"] for g in report["safety_gates"])
    report["cases"] = results
    return report


def safety_gates(report: dict[str, Any]) -> list[dict[str, Any]]:
    db = report["database_safety"]
    status = report["status_safety"]
    gates = [
        ("unsupported-code rate == 0", db["unsupported_code_rate"]),
        ("cross-version contamination == 0", db["cross_version_contamination"]),
        ("coding-system contamination == 0", db["coding_system_contamination"]),
        ("non-READY dataset leakage == 0", db["non_ready_leakage"]),
        ("fatal coding-rule violations == 0", report["rules"]["fatal_rule_violations"]),
        ("invalid rule target codes == 0", report["rules"]["invalid_target_codes"]),
        (
            "unsupported specificity rate == 0",
            report["specificity"]["unsupported_specificity_rate"] or 0,
        ),
        ("negated/ruled-out condition returned as active == 0", status.get("negated_as_active", 0)),
        (
            "family-history condition returned as active == 0",
            status.get("family_history_as_active", 0),
        ),
        ("personal-history condition returned as active == 0", status.get("history_as_active", 0)),
        (
            "must-not-return code returned == 0",
            report["final_selection"]["must_not_return_violations"],
        ),
        ("fabricated evidence == 0", report["evidence"]["fabricated_evidence"]),
        ("exclusion/incorrect evidence == 0", report["evidence"]["incorrect_evidence"]),
        (
            "dataset resolution errors == 0",
            len(db["dataset_resolution_errors"]),
        ),
    ]
    return [{"gate": name, "value": value, "passed": not value} for name, value in gates]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    """Concise human-readable summary (the JSON report holds every detail)."""
    lines = [
        "# Clinical evaluation report (synthetic)",
        "",
        f"> {report['meta']['disclaimer']}",
        "",
        f"- Scenarios: {report['dataset']['scenarios']} "
        f"(abstention: {report['dataset']['abstention_scenarios']})",
        f"- Embedding: {report['meta'].get('embedding_provider')} / "
        f"{report['meta'].get('embedding_model')}; semantic status "
        f"{report['retrieval']['semantic_status']}",
        f"- Passed scenarios: {report['final_selection']['scenarios_passed']}/"
        f"{report['final_selection']['scenarios']}",
        f"- **Safety gates: {'PASS' if report['safety_passed'] else 'FAIL'}**",
        "",
        "## Safety gates",
        "",
        "| Gate | Value | Result |",
        "|---|---|---|",
    ]
    lines += [
        f"| {g['gate']} | {_fmt(g['value'])} | {'PASS' if g['passed'] else 'FAIL'} |"
        for g in report["safety_gates"]
    ]
    c = report["concept_extraction"]
    lines += [
        "",
        "## Concept extraction",
        "",
        f"precision {_fmt(c['precision'])}, recall {_fmt(c['recall'])}, assertion accuracy "
        f"{_fmt(c['assertion_accuracy'])}, attributes {_fmt(c['attribute_accuracy_overall'])}",
        "",
        "## Retrieval (each method alone)",
        "",
        "| Mode | R@1 | R@3 | R@5 | R@10 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for mode, s in report["retrieval"]["by_mode"].items():
        lines.append(
            f"| {mode} | {_fmt(s['recall@1'])} | {_fmt(s['recall@3'])} | {_fmt(s['recall@5'])} "
            f"| {_fmt(s['recall@10'])} | {_fmt(s['mrr'])} |"
        )
    rr = report["reranking"]
    sel = report["final_selection"]
    sp = report["specificity"]
    ab = report["abstention"]
    lines += [
        "",
        "## Reranking and selection",
        "",
        f"top-1 before {_fmt(rr['top1_before'])} -> after {_fmt(rr['top1_after'])}; improved "
        f"{rr['improved']}, degraded {rr['degraded']}, correct removed "
        f"{rr['correct_candidate_removed']}, incorrect promoted "
        f"{rr['incorrect_candidate_promoted']}",
        f"expected-code recall {_fmt(sel['expected_code_recall'])}, code precision "
        f"{_fmt(sel['code_precision'])}, exact code-set match {_fmt(sel['exact_code_set_match'])}",
        "",
        "## Specificity and abstention",
        "",
        f"unsupported specificity rate {_fmt(sp['unsupported_specificity_rate'])}; safe fallback "
        f"{_fmt(sp['safe_fallback_rate'])}; missing-information recall "
        f"{_fmt(sp['missing_information_recall'])}, precision "
        f"{_fmt(sp['missing_information_precision'])}",
        f"abstention precision {_fmt(ab['precision'])}, recall {_fmt(ab['recall'])}, "
        f"false-positive code rate {_fmt(ab['false_positive_code_rate'])}",
        "",
        "## Failures by pipeline stage",
        "",
    ]
    for stage, count in report["failures"]["by_primary_stage"].items():
        lines.append(f"- {stage}: {count}")
    lines += ["", "## Latency (ms, synthetic environment)", "", "| Stage | mean | p50 | p95 |"]
    lines.append("|---|---|---|---|")
    for stage, s in report["latency_ms"].items():
        lines.append(f"| {stage} | {s['mean']} | {s['p50']} | {s['p95']} |")
    return "\n".join(lines) + "\n"
