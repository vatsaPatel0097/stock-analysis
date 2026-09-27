"""Calibration and realised-loss report from resolved calls.

Joins the append-only calls table to the outcomes side table in memory.
Does not rewrite historical calls and does not recompute outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.resolve import list_outcomes
from store.calls import DEFAULT_PATH, list_calls

DISCLAIMER = (
    "Educational research only. Not investment advice. "
    "Past patterns do not predict future results."
)

# Half-open buckets: [0.35, 0.45), …, [0.95, 1.05).
_BUCKET_EDGES = [round(0.35 + 0.10 * i, 2) for i in range(8)]  # 0.35 … 1.05


def build_report(path: Path | str | None = None) -> dict:
    """Join calls to outcomes and return the calibration + loss summary."""
    db_path = Path(path) if path is not None else DEFAULT_PATH
    calls = {int(row["id"]): row for row in list_calls(db_path)}
    outcomes = list_outcomes(db_path)

    joined: list[dict] = []
    for row in outcomes:
        call = calls.get(int(row["call_id"]))
        if call is None:
            continue
        joined.append(
            {
                "call_id": int(row["call_id"]),
                "action": call["action"],
                "payload": call["payload"],
                "outcome": row["outcome"],
                "realised_pct": float(row["realised_pct"]),
                "worst_adverse_pct": float(row["worst_adverse_pct"]),
                "predicted_prob": _predicted_prob(call["payload"]),
            }
        )

    counts = _count_outcomes(joined)
    buy_rows = [row for row in joined if row["action"] == "BUY"]
    buy_counts = _count_outcomes(buy_rows)
    buy_wins = buy_counts["target"]
    buy_n = len(buy_rows)
    buy_win_rate = (buy_wins / buy_n) if buy_n else None

    buckets = _calibration_buckets(joined)
    errors = [bucket["abs_error"] for bucket in buckets if bucket["n"] > 0]
    mean_abs_error = (sum(errors) / len(errors)) if errors else None

    stop_rows = [row for row in joined if row["outcome"] == "stop"]
    loss_dist = _loss_distribution(stop_rows)

    return {
        "resolved": len(joined),
        "counts": counts,
        "buy": {
            "n": buy_n,
            "counts": buy_counts,
            "win_rate": buy_win_rate,
        },
        "calibration": {
            "buckets": buckets,
            "mean_abs_error": mean_abs_error,
        },
        "loss_distribution": loss_dist,
        "disclaimer": DISCLAIMER,
    }


def _predicted_prob(payload: dict) -> float | None:
    """Recommended rung probability, else user_target.prob."""
    for row in payload.get("targets") or []:
        if row.get("verdict") == "recommended":
            try:
                return float(row["prob"])
            except (TypeError, ValueError, KeyError):
                return None
    user = payload.get("user_target")
    if isinstance(user, dict) and user.get("prob") is not None:
        try:
            return float(user["prob"])
        except (TypeError, ValueError):
            return None
    return None


def _count_outcomes(rows: list[dict]) -> dict[str, int]:
    counts = {"target": 0, "stop": 0, "neither": 0}
    for row in rows:
        label = row["outcome"]
        if label in counts:
            counts[label] += 1
    return counts


def _calibration_buckets(rows: list[dict]) -> list[dict]:
    buckets: list[dict] = []
    for i in range(len(_BUCKET_EDGES) - 1):
        low = _BUCKET_EDGES[i]
        high = _BUCKET_EDGES[i + 1]
        members = [
            row
            for row in rows
            if row["predicted_prob"] is not None and low <= float(row["predicted_prob"]) < high
        ]
        n = len(members)
        if n == 0:
            buckets.append(
                {
                    "lo": low,
                    "hi": high,
                    "n": 0,
                    "predicted_mean": None,
                    "actual_win_rate": None,
                    "abs_error": None,
                }
            )
            continue
        predicted_mean = sum(float(row["predicted_prob"]) for row in members) / n
        wins = sum(1 for row in members if row["outcome"] == "target")
        actual = wins / n
        buckets.append(
            {
                "lo": low,
                "hi": high,
                "n": n,
                "predicted_mean": predicted_mean,
                "actual_win_rate": actual,
                "abs_error": abs(predicted_mean - actual),
            }
        )
    return buckets


def _loss_distribution(stop_rows: list[dict]) -> dict:
    n = len(stop_rows)
    if n == 0:
        return {
            "n": 0,
            "at_or_better_than_2pct": None,
            "between_2_and_2_5pct": None,
            "worse_than_2_5pct": None,
            "worst_worse_than_fill": None,
        }

    at_or_better = 0
    between = 0
    worse = 0
    worst_worse = 0
    for row in stop_rows:
        realised = float(row["realised_pct"])
        # realised is negative on a stop. −0.02 is exactly 2%.
        loss_mag = -realised
        if loss_mag <= 0.02:
            at_or_better += 1
        elif loss_mag <= 0.025:
            between += 1
        else:
            worse += 1
        fill_loss = -realised
        worst = float(row["worst_adverse_pct"])
        if worst > fill_loss + 1e-12:
            worst_worse += 1

    return {
        "n": n,
        "at_or_better_than_2pct": at_or_better / n,
        "between_2_and_2_5pct": between / n,
        "worse_than_2_5pct": worse / n,
        "worst_worse_than_fill": worst_worse / n,
    }


def main() -> None:
    print(json.dumps(build_report(), indent=2, default=str))


if __name__ == "__main__":
    main()
