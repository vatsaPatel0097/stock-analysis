"""One-box Streamlit UI. Renders analyze() JSON only — no math in the UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze import AmbiguousQuery, analyze
from data.prices import PriceError
from data.resolve import ResolveError
from ui import view

_THEME = Path(__file__).with_name("theme.css")


def _load_theme() -> str:
    return _THEME.read_text(encoding="utf-8")


def _fixture_kwargs():
    """Optional offline fixtures when SSA_FIXTURE=1 (browser smoke / demos)."""
    if os.environ.get("SSA_FIXTURE") != "1":
        return {}
    import pandas as pd

    def _frame(n: int, *, step: float = 0.25, spread: float = 0.01):
        rows = []
        close = 100.0
        for _ in range(n):
            rows.append((close, close * (1.0 + spread), close * (1.0 - spread), 2_000_000.0))
            close += step
        index = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        return pd.DataFrame(
            {
                "Open": [row[0] for row in rows],
                "High": [row[1] for row in rows],
                "Low": [row[2] for row in rows],
                "Close": [row[0] for row in rows],
                "Volume": [row[3] for row in rows],
            },
            index=index,
        )

    buy = _frame(80)
    skip = _frame(30)

    def price_loader(ticker: str, *, as_of=None):
        # Short history → sample-size SKIP. Long history → BUY path.
        key = ticker.upper()
        if key in {"WIPRO.NS", "HCLTECH.NS"}:
            return skip.copy()
        return buy.copy()

    def searcher(query: str):
        return []

    return {"price_loader": price_loader, "searcher": searcher}


def _run_analyze(query: str, user_target: float | None):
    kwargs = _fixture_kwargs()
    return analyze(query, user_target=user_target, **kwargs)


def _render_card(payload: dict) -> None:
    price = payload["price"]
    banner = view.action_banner(payload)
    action = banner["action"].lower()
    css = f"ssa-action ssa-action-{action}" if action in {"buy", "wait", "skip"} else "ssa-action"

    st.markdown('<div class="ssa-card">', unsafe_allow_html=True)
    st.markdown(
        f"**{payload['query']}** · `{payload['ticker']}`  \n"
        f"<span class='ssa-price-current'>{view.format_rupees(price['current'])}</span>  \n"
        f"<span class='ssa-muted'>High {view.format_rupees(price['day_high'])} · "
        f"Low {view.format_rupees(price['day_low'])} · "
        f"Prev {view.format_rupees(price['prev_close'])} · "
        f"As of {payload['as_of']}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='{css}'>{banner['text']}</div>", unsafe_allow_html=True)

    cells = view.plan_cells(payload)
    cols = st.columns(4)
    for column, (label, value) in zip(cols, cells.items()):
        column.markdown(
            f"<div class='ssa-plan-cell'><div class='ssa-plan-label'>{label}</div>"
            f"<div class='ssa-plan-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

    rows = view.target_table_rows(payload)
    if rows:
        st.markdown("#### Target ladder")
        st.dataframe(rows, hide_index=True, width="stretch")

    user = view.user_target_row(payload)
    if user is not None:
        st.markdown("#### Your target")
        st.dataframe([user], hide_index=True, width="stretch")

    chips = view.regime_chips(payload)
    if chips:
        chip_html = " ".join(f"<span class='ssa-chip'>{text}</span>" for text in chips)
        st.markdown(f"<div class='ssa-chips'>{chip_html}</div>", unsafe_allow_html=True)

    if payload.get("news"):
        news = payload["news"]
        st.markdown(f"**News** ({news.get('sentiment', '')}): {news.get('summary', '')}")
    if payload.get("explanation"):
        st.markdown(payload["explanation"])

    st.markdown(f"<p class='ssa-gap'>{view.card_gap_risk()}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Stock Swing Advisor", layout="centered")
    st.markdown(f"<style>{_load_theme()}</style>", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.session_state.result = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "picks" not in st.session_state:
        st.session_state.picks = None
    if "loading" not in st.session_state:
        st.session_state.loading = False

    st.markdown('<div class="ssa-shell">', unsafe_allow_html=True)
    st.markdown('<p class="ssa-title">Stock Swing Advisor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="ssa-subtitle">NSE research · not investment advice</p>',
        unsafe_allow_html=True,
    )

    with st.form("search"):
        query = st.text_input(
            "Stock",
            placeholder="Stock name or ticker — e.g. Reliance",
            label_visibility="collapsed",
        )
        target_raw = st.text_input("Wished target (₹), optional", value="")
        submitted = st.form_submit_button("Analyze", type="primary", disabled=st.session_state.loading)

    if submitted:
        st.session_state.result = None
        st.session_state.error = None
        st.session_state.picks = None
        trimmed = (query or "").strip()
        if not trimmed:
            st.session_state.error = view.EMPTY_ERROR
        else:
            wished = None
            if target_raw.strip():
                try:
                    wished = float(target_raw.strip())
                except ValueError:
                    st.session_state.error = "Wished target must be a number of rupees."
                    wished = False
            if wished is not False:
                st.session_state.loading = True
                try:
                    st.session_state.result = _run_analyze(trimmed, wished)
                except AmbiguousQuery as exc:
                    st.session_state.picks = view.pick_list_rows(exc.matches)
                except (ResolveError, PriceError, ValueError) as exc:
                    st.session_state.error = view.error_message(str(exc))
                finally:
                    st.session_state.loading = False

    if st.session_state.loading:
        st.spinner("Analyzing…")
        st.info("Analyzing…")
    elif st.session_state.error:
        st.markdown(
            f"<div class='ssa-error'>{view.error_message(st.session_state.error)}</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.picks is not None:
        st.markdown(
            "<div class='ssa-pick-heading'>Several matches. Pick one.</div>",
            unsafe_allow_html=True,
        )
        for row in st.session_state.picks:
            label = f"{row['name']}  {row['ticker']}"
            if st.button(label, key=f"pick-{row['ticker']}"):
                try:
                    st.session_state.result = _run_analyze(row["ticker"], None)
                    st.session_state.picks = None
                    st.session_state.error = None
                except (ResolveError, PriceError, ValueError) as exc:
                    st.session_state.error = view.error_message(str(exc))
                    st.session_state.result = None
                st.rerun()
    elif st.session_state.result is not None:
        _render_card(st.session_state.result)
    else:
        st.markdown(f"<p class='ssa-empty'>{view.empty_message()}</p>", unsafe_allow_html=True)

    st.markdown(
        f"<p class='ssa-disclaimer'>{view.card_disclaimer()}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
