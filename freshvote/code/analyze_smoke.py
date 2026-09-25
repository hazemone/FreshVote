"""Typed aggregation of the non-publication FreshVote smoke-test output."""

import csv
from collections import defaultdict
from pathlib import Path


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    source = root / "results" / "smoke" / "smoke_summary.csv"
    destination = root / "results" / "smoke" / "smoke_policy_summary.csv"
    grouped = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["policy"]].append(row)

    output = []
    for policy, rows in sorted(grouped.items()):
        n = len(rows)
        output.append({
            "policy": policy,
            "runs": n,
            "stable_correct_rate": sum(as_bool(row["stable_correct_consensus"]) for row in rows) / n,
            "transient_rate": sum(as_bool(row["transient_consensus"]) for row in rows) / n,
            "mean_fresh_message_acceptance": sum(float(row["fresh_message_acceptance"]) for row in rows) / n,
            "total_age_rejections": sum(int(row["rejected_age_messages"]) for row in rows),
        })

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    for row in output:
        print(
            f"{row['policy']}: runs={row['runs']}, "
            f"stable_correct={row['stable_correct_rate']:.3f}, "
            f"transient={row['transient_rate']:.3f}, "
            f"acceptance={row['mean_fresh_message_acceptance']:.3f}, "
            f"rejections={row['total_age_rejections']}"
        )


if __name__ == "__main__":
    main()
