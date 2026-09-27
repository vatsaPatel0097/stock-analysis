"""Pure formatters for the one-box UI. No analytics and no gates."""

from __future__ import annotations

from rules.gates import gap_risk_note

EMPTY_COPY = "Type a name. Get current price, buy, stop, targets."
EMPTY_ERROR = "Type a stock name or ticker."
DISCLAIMER = (
    "Educational research only. Not investment advice. "
    "Past patterns do not predict future results."
)


def empty_message() -> str:
    return EMPTY_COPY


def error_message(text: str) -> str:
    sentence = " ".join(str(text).strip().split())
    if not sentence:
        return EMPTY_ERROR
    return sentence


def pick_list_rows(matches) -> list[dict[str, str]]:
    """Rows for the pick list. No prices."""
    rows: list[dict[str, str]] = []
    for item in matches:
        name = getattr(item, "name", None)
        ticker = getattr(item, "ticker", None)
        if isinstance(item, dict):
            name = item.get("name")
            ticker = item.get("ticker")
        if not name or not ticker:
            continue
        rows.append({"name": str(name), "ticker": str(ticker)})
    return rows


def format_rupees(value) -> str:
    if value is None:
        return "—"
    return f"₹{float(value):.2f}"


def format_percent(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}%"


def format_prob(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def format_days(row: dict) -> str:
    band = row.get("days_p25_p75")
    if band:
        return f"{band[0]}–{band[1]}"
    if row.get("days_median") is None:
        return "—"
    return str(row["days_median"])


def plan_cells(payload: dict) -> dict[str, str]:
    plan = payload.get("plan") or {}
    action = plan.get("action")
    recommended = plan.get("recommended_target")
    recommended_pct = plan.get("recommended_target_pct")
    sell_by_day = plan.get("sell_by_day")
    sell_by_date = plan.get("sell_by_date")

    if action == "SKIP" and recommended is None:
        target_text = "—"
        sell_text = "—"
    else:
        target_text = format_rupees(recommended)
        if recommended_pct is not None and recommended is not None:
            target_text = f"{target_text} (+{format_percent(recommended_pct).rstrip('%')}%)"
        if sell_by_day is None:
            sell_text = "—"
        else:
            sell_text = f"day {sell_by_day}"
            if sell_by_date:
                sell_text = f"{sell_text} · {sell_by_date}"

    stop = format_rupees(plan.get("stop_loss"))
    stop_pct = plan.get("stop_pct")
    if stop_pct is not None and plan.get("stop_loss") is not None:
        stop = f"{stop} ({format_percent(stop_pct)})"

    return {
        "Buy at": format_rupees(plan.get("entry")),
        "Stop": stop,
        "Recommended target": target_text,
        "Sell by": sell_text,
    }


def target_table_rows(payload: dict) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in payload.get("targets") or []:
        rows.append(
            {
                "Target": format_rupees(row.get("price")),
                "Gain": f"+{format_percent(row.get('pct')).rstrip('%')}%",
                "Probability": format_prob(row.get("prob")),
                "Typical days": format_days(row),
                "Verdict": str(row.get("verdict") or ""),
            }
        )
    return rows


def user_target_row(payload: dict) -> dict[str, str] | None:
    row = payload.get("user_target")
    if not row:
        return None
    return {
        "Target": format_rupees(row.get("price")),
        "Gain": f"+{format_percent(row.get('pct')).rstrip('%')}%",
        "Probability": format_prob(row.get("prob")),
        "Typical days": format_days(row),
        "Verdict": str(row.get("verdict") or ""),
    }


def regime_chips(payload: dict) -> list[str]:
    regime = payload.get("regime") or {}
    chips: list[str] = []
    if regime.get("trend") is not None:
        chips.append(f"trend {regime['trend']}")
    if regime.get("rsi14") is not None:
        chips.append(f"RSI {regime['rsi14']}")
    if regime.get("atr_pct") is not None:
        chips.append(f"ATR {format_percent(regime['atr_pct'])}")
    above = regime.get("above_200dma")
    if above is True:
        chips.append("above 200-DMA")
    elif above is False:
        chips.append("below 200-DMA")
    return chips


def action_banner(payload: dict) -> dict[str, str]:
    plan = payload.get("plan") or {}
    action = str(plan.get("action") or "")
    reason = payload.get("skip_reason")
    text = action
    if action == "SKIP" and reason:
        text = f"{action} — {reason}"
    return {"action": action, "text": text}


def model_line(payload: dict) -> str | None:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        return None
    return f"Model: {model.strip()}"


def card_gap_risk() -> str:
    return gap_risk_note()


def card_disclaimer() -> str:
    return DISCLAIMER


def watchlist_line(row: dict) -> str:
    """One memory line from stored snapshot + mark fields. No math."""
    ticker = str(row.get("ticker") or "")
    added = format_rupees(row.get("price_at_add"))
    action = str(row.get("action") or "")
    entry = format_rupees(row.get("entry"))
    stop = format_rupees(row.get("stop_loss"))
    target = format_rupees(row.get("recommended_target"))
    sell_day = row.get("sell_by_day")
    sell_date = row.get("sell_by_date")
    if sell_day is None:
        sell_by = "—"
    else:
        sell_by = f"day {sell_day}"
        if sell_date:
            sell_by = f"{sell_by} · {sell_date}"

    left = (
        f"{ticker} · added at {added} · we said {action} · "
        f"entry {entry} · stop {stop} · target {target} · sell by {sell_by}"
    )

    latest = row.get("latest_price")
    if latest is None:
        return left

    change = row.get("change_pct")
    now = format_rupees(latest)
    if change is None:
        right = f"now {now}"
    else:
        sign = "+" if float(change) > 0 else ""
        right = f"now {now} ({sign}{format_percent(change)})"

    outcome = row.get("outcome")
    if outcome:
        fill = row.get("fill_price")
        exit_day = row.get("exit_day")
        outcome_bits = [str(outcome)]
        if exit_day is not None:
            outcome_bits.append(f"day {exit_day}")
        if fill is not None:
            outcome_bits.append(format_rupees(fill))
        right = f"{right} · outcome {' · '.join(outcome_bits)}"

    return f"{left} · {right}"
