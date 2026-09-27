#!/usr/bin/env python3
"""Validate the frozen FreshVote primary-evaluation dataset without modifying raw inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_COLUMNS = [
    "accepted_message_mean_age",
    "accepted_messages",
    "agents",
    "collapse_detected",
    "collapse_timestep",
    "consensus_residence_ticks",
    "delay",
    "delivered_messages",
    "final_alive",
    "final_correct_fraction",
    "final_fiedler",
    "final_fraction_one",
    "final_fragmented",
    "final_ghost",
    "final_mobile",
    "final_order_param",
    "final_variance",
    "fresh_message_acceptance",
    "freshness_adaptive_kappa",
    "freshness_fixed_decay_lambda",
    "freshness_fixed_ttl",
    "jitter",
    "loss",
    "policy",
    "post_hit_loss",
    "rejected_age_messages",
    "seed",
    "stable_correct_consensus",
    "stable_decision_time",
    "stable_incorrect_consensus",
    "stable_outcome",
    "steps",
    "total_steps_recorded",
    "transient_consensus",
]
POLICIES = ("baseline", "hard_ttl", "fixed_decay", "adaptive_freshvote")
DELAYS = (0, 2, 4, 6)
JITTERS = (0, 2, 4)
LOSSES = (0.0, 0.1)
SEEDS = tuple(range(21, 61))
EXPECTED_ROWS = len(POLICIES) * len(DELAYS) * len(JITTERS) * len(LOSSES) * len(SEEDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid Boolean {value!r}")


def parse_int(value: str, *, optional: bool = False) -> int | None:
    if optional and value == "":
        return None
    if value == "":
        raise ValueError("missing integer")
    return int(value)


def parse_float(value: str, *, optional: bool = False) -> float | None:
    if optional and value == "":
        return None
    if value == "":
        raise ValueError("missing float")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite float {value!r}")
    return parsed


def approx(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", dest="csv_path", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()

    def add_error(category: str, detail: str, row_number: int | None = None) -> None:
        issue_counts[category] += 1
        if sum(1 for item in errors if item["category"] == category) < 8:
            item: dict[str, Any] = {"category": category, "detail": detail}
            if row_number is not None:
                item["row_number"] = row_number
            errors.append(item)

    def add_warning(category: str, detail: str) -> None:
        warnings.append({"category": category, "detail": detail})

    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    expected_manifest = {
        "agents": 100,
        "calibration_seed_range": [1, 20],
        "delays": list(DELAYS),
        "diagnostic_limit": 0,
        "evaluation_seed_range": [21, 60],
        "executed_size": EXPECTED_ROWS,
        "expected_full_size": EXPECTED_ROWS,
        "frozen_policies": [
            {"policy": "baseline"},
            {"fixed_ttl": 9, "policy": "hard_ttl"},
            {"fixed_decay_lambda": 0.35, "policy": "fixed_decay"},
            {"adaptive_kappa": 1.5, "policy": "adaptive_freshvote"},
        ],
        "jitters": list(JITTERS),
        "losses": list(LOSSES),
        "purpose": "primary held-out FreshVote evaluation",
        "steps": 3000,
    }
    if manifest != expected_manifest:
        add_error("manifest_mismatch", "Manifest differs from the exact frozen primary-evaluation specification.")

    row_keys: Counter[tuple[str, int, int, float, int]] = Counter()
    cell_counts: Counter[tuple[str, int, int, float]] = Counter()
    outcome_counts: Counter[str] = Counter()
    policy_outcomes: Counter[tuple[str, str]] = Counter()
    rows: list[dict[str, str]] = []

    with args.csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            add_error(
                "csv_schema_mismatch",
                f"Observed columns {reader.fieldnames!r}; expected {EXPECTED_COLUMNS!r}.",
            )
        for row_number, row in enumerate(reader, start=2):
            rows.append(row)
            try:
                policy = row["policy"]
                delay = parse_int(row["delay"])
                jitter = parse_int(row["jitter"])
                loss = parse_float(row["loss"])
                seed = parse_int(row["seed"])
                agents = parse_int(row["agents"])
                steps = parse_int(row["steps"])
                recorded = parse_int(row["total_steps_recorded"])
                stable_correct = parse_bool(row["stable_correct_consensus"])
                stable_incorrect = parse_bool(row["stable_incorrect_consensus"])
                transient = parse_bool(row["transient_consensus"])
                post_hit_loss = parse_bool(row["post_hit_loss"])
                collapse = parse_bool(row["collapse_detected"])
                fragmented = parse_bool(row["final_fragmented"])
                collapse_time = parse_int(row["collapse_timestep"], optional=True)
                decision_time = parse_int(row["stable_decision_time"], optional=True)
                accepted = parse_int(row["accepted_messages"])
                rejected = parse_int(row["rejected_age_messages"])
                delivered = parse_int(row["delivered_messages"])
                residence = parse_int(row["consensus_residence_ticks"])
                final_alive = parse_int(row["final_alive"])
                final_ghost = parse_int(row["final_ghost"])
                final_mobile = parse_int(row["final_mobile"])
                accepted_age = parse_float(row["accepted_message_mean_age"], optional=True)
                acceptance = parse_float(row["fresh_message_acceptance"])
                final_correct_fraction = parse_float(row["final_correct_fraction"])
                final_fraction_one = parse_float(row["final_fraction_one"])
                final_order = parse_float(row["final_order_param"])
                final_variance = parse_float(row["final_variance"])
                final_fiedler = parse_float(row["final_fiedler"])
                adaptive_kappa = parse_float(row["freshness_adaptive_kappa"], optional=True)
                fixed_lambda = parse_float(row["freshness_fixed_decay_lambda"], optional=True)
                fixed_ttl = parse_int(row["freshness_fixed_ttl"], optional=True)
                _ = transient, post_hit_loss, fragmented
            except (KeyError, TypeError, ValueError) as exc:
                add_error("malformed_row", str(exc), row_number)
                continue

            if policy not in POLICIES:
                add_error("unexpected_policy", policy, row_number)
            if delay not in DELAYS or jitter not in JITTERS or loss not in LOSSES or seed not in SEEDS:
                add_error(
                    "unexpected_factor_level",
                    f"policy={policy}, delay={delay}, jitter={jitter}, loss={loss}, seed={seed}",
                    row_number,
                )

            key = (policy, int(delay), int(jitter), float(loss), int(seed))
            row_keys[key] += 1
            cell_counts[key[:-1]] += 1

            if agents != 100 or steps != 3000 or recorded != 3000:
                add_error(
                    "run_size_mismatch",
                    f"agents={agents}, steps={steps}, total_steps_recorded={recorded}",
                    row_number,
                )

            expected_parameters = {
                "baseline": (None, None, None),
                "hard_ttl": (None, None, 9),
                "fixed_decay": (None, 0.35, None),
                "adaptive_freshvote": (1.5, None, None),
            }
            expected_kappa, expected_lambda, expected_ttl = expected_parameters.get(policy, (None, None, None))
            if adaptive_kappa != expected_kappa or fixed_lambda != expected_lambda or fixed_ttl != expected_ttl:
                add_error(
                    "policy_parameter_mismatch",
                    f"policy={policy}, kappa={adaptive_kappa}, lambda={fixed_lambda}, ttl={fixed_ttl}",
                    row_number,
                )

            if stable_correct and stable_incorrect:
                add_error("mutually_exclusive_outcomes", "Both stable outcome flags are true.", row_number)
            expected_outcome = "correct" if stable_correct else "incorrect" if stable_incorrect else "none"
            if row["stable_outcome"] != expected_outcome:
                add_error(
                    "stable_outcome_mismatch",
                    f"flags imply {expected_outcome!r}, field is {row['stable_outcome']!r}",
                    row_number,
                )
            if (stable_correct or stable_incorrect) != (decision_time is not None):
                add_error(
                    "decision_time_presence_mismatch",
                    f"stable_correct={stable_correct}, stable_incorrect={stable_incorrect}, time={decision_time}",
                    row_number,
                )
            if decision_time is not None and not 0 <= decision_time < 3000:
                add_error("decision_time_out_of_range", str(decision_time), row_number)

            outcome_counts[expected_outcome] += 1
            policy_outcomes[(policy, expected_outcome)] += 1

            if min(accepted, rejected, delivered, residence, final_alive, final_ghost, final_mobile) < 0:
                add_error("negative_count", "One or more count fields are negative.", row_number)
            if accepted + rejected != delivered:
                add_error(
                    "message_accounting_mismatch",
                    f"accepted={accepted}, rejected={rejected}, delivered={delivered}",
                    row_number,
                )
            expected_acceptance = accepted / delivered if delivered else 0.0
            if not approx(acceptance, expected_acceptance, tolerance=1e-10):
                add_error(
                    "acceptance_ratio_mismatch",
                    f"reported={acceptance}, calculated={expected_acceptance}",
                    row_number,
                )
            if accepted > 0 and accepted_age is None:
                add_error("missing_accepted_age", "Accepted messages exist but mean age is missing.", row_number)
            if accepted_age is not None and accepted_age < 0:
                add_error("negative_accepted_age", str(accepted_age), row_number)
            if residence > 3000:
                add_error("residence_out_of_range", str(residence), row_number)
            if final_alive > 100 or final_ghost + final_mobile != final_alive:
                add_error(
                    "agent_accounting_mismatch",
                    f"alive={final_alive}, ghost={final_ghost}, mobile={final_mobile}",
                    row_number,
                )
            for field_name, value in (
                ("fresh_message_acceptance", acceptance),
                ("final_correct_fraction", final_correct_fraction),
                ("final_fraction_one", final_fraction_one),
                ("final_order_param", final_order),
                ("final_variance", final_variance),
            ):
                if not 0.0 <= value <= 1.0:
                    add_error("fraction_out_of_range", f"{field_name}={value}", row_number)
            if final_fiedler < -1e-10:
                add_error("negative_fiedler", str(final_fiedler), row_number)
            if collapse != (collapse_time is not None):
                add_error(
                    "collapse_time_presence_mismatch",
                    f"collapse={collapse}, collapse_timestep={collapse_time}",
                    row_number,
                )
            if collapse_time is not None and not 0 <= collapse_time < 3000:
                add_error("collapse_time_out_of_range", str(collapse_time), row_number)

    if len(rows) != EXPECTED_ROWS:
        add_error("row_count_mismatch", f"Observed {len(rows)} data rows; expected {EXPECTED_ROWS}.")

    expected_keys = {
        (policy, delay, jitter, loss, seed)
        for policy, delay, jitter, loss, seed in itertools.product(POLICIES, DELAYS, JITTERS, LOSSES, SEEDS)
    }
    observed_keys = set(row_keys)
    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    duplicate_keys = sorted((key, count) for key, count in row_keys.items() if count != 1)
    if missing_keys:
        add_error("missing_run_keys", f"Missing {len(missing_keys)} keys; first={missing_keys[:5]!r}")
    if unexpected_keys:
        add_error("unexpected_run_keys", f"Unexpected {len(unexpected_keys)} keys; first={unexpected_keys[:5]!r}")
    if duplicate_keys:
        add_error("duplicate_run_keys", f"Non-unique {len(duplicate_keys)} keys; first={duplicate_keys[:5]!r}")

    expected_cells = {
        (policy, delay, jitter, loss)
        for policy, delay, jitter, loss in itertools.product(POLICIES, DELAYS, JITTERS, LOSSES)
    }
    bad_cells = sorted((cell, cell_counts[cell]) for cell in expected_cells if cell_counts[cell] != 40)
    if bad_cells:
        add_error("cell_size_mismatch", f"Cells not equal to 40: {bad_cells[:10]!r}")

    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    progress_pattern = re.compile(
        r"\[(\d+)/3840\]\s+([a-z_]+)\s+d=(\d+)\s+j=(\d+)\s+p=(0\.0|0\.1)\s+seed=(\d+)"
    )
    progress_matches = progress_pattern.findall(log_text)
    progress_indices = Counter(int(item[0]) for item in progress_matches)
    progress_keys = Counter(
        (item[1], int(item[2]), int(item[3]), float(item[4]), int(item[5])) for item in progress_matches
    )
    if len(progress_matches) != EXPECTED_ROWS:
        add_error("log_progress_count_mismatch", f"Parsed {len(progress_matches)} progress entries.")
    if set(progress_indices) != set(range(1, EXPECTED_ROWS + 1)) or any(count != 1 for count in progress_indices.values()):
        add_error("log_progress_index_mismatch", "Progress indices are missing or duplicated.")
    if progress_keys != row_keys:
        add_error("log_csv_key_mismatch", "Logged policy-condition-seed jobs do not exactly match CSV keys.")
    if "Completed 3840 of 3840 planned evaluation runs" not in log_text:
        add_error("missing_completion_marker", "Full completion marker is absent from the log.")
    for marker in ("Traceback", "Exception", "Killed"):
        if marker.lower() in log_text.lower():
            add_error("runtime_failure_marker", f"Found {marker!r} in log.")

    add_warning(
        "environment_record_not_in_archive",
        "The archive does not contain Python/NumPy versions or pip freeze; request them from the execution machine for the reproducibility record.",
    )

    cell_path = args.output_dir / "factor_cell_counts.csv"
    with cell_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "delay", "jitter", "loss", "runs"])
        for cell in sorted(expected_cells):
            writer.writerow([*cell, cell_counts[cell]])

    validation = {
        "status": "PASS" if not issue_counts else "FAIL",
        "expected_rows": EXPECTED_ROWS,
        "observed_rows": len(rows),
        "unique_run_keys": len(row_keys),
        "factor_cells": len(cell_counts),
        "factor_cells_with_40_runs": sum(count == 40 for count in cell_counts.values()),
        "manifest_exact_match": manifest == expected_manifest,
        "log_progress_entries": len(progress_matches),
        "log_csv_exact_key_match": progress_keys == row_keys,
        "completion_marker_present": "Completed 3840 of 3840 planned evaluation runs" in log_text,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "policy_outcome_counts": {
            policy: {outcome: policy_outcomes[(policy, outcome)] for outcome in ("correct", "incorrect", "none")}
            for policy in POLICIES
        },
        "sha256": {
            "manifest": sha256(args.manifest),
            "evaluation_csv": sha256(args.csv_path),
            "execution_log": sha256(args.log),
        },
        "issue_counts": dict(sorted(issue_counts.items())),
        "error_samples": errors,
        "warnings": warnings,
    }
    json_path = args.output_dir / "validation_report.json"
    json_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# FreshVote primary-evaluation validation",
        "",
        f"**Status:** {validation['status']}",
        "",
        f"- Data rows: {len(rows):,} / {EXPECTED_ROWS:,}",
        f"- Unique policy-condition-seed keys: {len(row_keys):,} / {EXPECTED_ROWS:,}",
        f"- Factor cells with 40 held-out seeds: {validation['factor_cells_with_40_runs']} / 96",
        f"- Frozen manifest exact match: {validation['manifest_exact_match']}",
        f"- Log entries reconciled to CSV: {validation['log_csv_exact_key_match']}",
        f"- Full completion marker present: {validation['completion_marker_present']}",
        "",
        "## Outcome counts",
        "",
        "| Policy | Stable correct | Stable incorrect | No stable outcome |",
        "|---|---:|---:|---:|",
    ]
    for policy in POLICIES:
        counts = validation["policy_outcome_counts"][policy]
        md_lines.append(f"| {policy} | {counts['correct']} | {counts['incorrect']} | {counts['none']} |")
    md_lines.extend(["", "## Integrity hashes", ""])
    for name, digest in validation["sha256"].items():
        md_lines.append(f"- `{name}`: `{digest}`")
    if issue_counts:
        md_lines.extend(["", "## Errors", ""])
        for category, count in sorted(issue_counts.items()):
            md_lines.append(f"- **{category}:** {count}")
    md_lines.extend([
        "",
        "## Reproducibility warning",
        "",
        warnings[0]["detail"],
        "",
        "The raw uploaded archive was not modified. This validation uses only the held-out primary-evaluation files.",
    ])
    (args.output_dir / "validation_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(validation, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
