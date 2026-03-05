"""
ui/tabs/atradeaday.py
=====================
Renders the A Trade A Day tab (tabs[8]).

Uses whatever data is already loaded in st.session_state.df from the sidebar.
Requires 1m, 2m, or 5m data — shows a clear error otherwise.
"""

import pandas as pd
import streamlit as st

from src.strategy.atradeaday import (
    run_atradeaday,
    ATradeADayParams,
    ATradeADayResults,
)
from ui.charts import create_equity_chart, create_price_chart_with_trades, PLOTLY_CONFIG


def render_atradeaday_tab() -> None:

    # ── Info banner ───────────────────────────────────────────────────────────
    with st.expander("ℹ️ How this strategy works", expanded=False):
        st.markdown("""
        **A Trade A Day** — one setup, once a day, full rules:

        1. **Mark the levels** — The first **5-minute candle** of the day (9:30 AM) sets
           the high and low. That's your entire analysis. This candle is used because it
           contains the highest volume of the trading day.
        2. **FVG breakout** — Wait for a 3-candle Fair Value Gap that breaks through one
           of those levels. The middle candle must cross with momentum, leaving a gap
           between candle 1 and candle 3's wicks.
        3. **Retest** — Price pulls back into the FVG gap zone.
        4. **Engulfing entry** — A candle engulfs the pullback candle. Enter at its close.
        5. **Exit** — SL at the first FVG candle's wick. TP at `RR × risk`. One trade. Walk away.

        ⚠️ **Requires 1m, 2m, or 5m data.** Load it from the sidebar before running.
        If 1m or 2m is loaded, the first 5-min candle is calculated automatically by resampling.
        """)

    # ── Data check ────────────────────────────────────────────────────────────
    if st.session_state.df is None:
        st.warning("⚠️ No data loaded. Use the sidebar to load a ticker with 1m, 2m, or 5m interval.")
        return

    df       = st.session_state.df
    info     = df.attrs if hasattr(df, 'attrs') else {}
    symbol   = info.get('symbol', 'Unknown')
    interval = info.get('interval', 'Unknown')
    n_bars   = len(df)
    n_days   = df.index.normalize().nunique()

    # Interval warning — shown before the run button so user sees it immediately
    valid_intervals = {'1m', '2m', '5m'}
    if interval not in valid_intervals:
        st.error(
            f"❌ Loaded data is **{interval}** — too coarse for this strategy. "
            f"A Trade A Day needs the first 5-minute candle of the day. "
            f"Please reload from the sidebar using **1m**, **2m**, or **5m** interval."
        )
        return

    st.info(
        f"Using: **{symbol}** · **{interval}** · **{n_bars:,} bars** · **{n_days} trading days**"
        + (" *(will be resampled to 5m for opening candle)*" if interval in {'1m', '2m'} else "")
    )

    st.divider()

    # ── Config row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

    risk_per_trade = c1.number_input(
        "Risk per trade ($)", min_value=10, max_value=50000,
        value=100, step=10, key="atad_risk",
    )
    rr_ratio = c2.number_input(
        "R:R ratio", min_value=1.0, max_value=10.0,
        value=3.0, step=0.5, key="atad_rr",
    )
    commission = c3.number_input(
        "Comm %", min_value=0.0, max_value=1.0,
        value=0.05, step=0.01, key="atad_commission",
    )

    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button(
            "🚀 Run", type="primary",
            use_container_width=True,
            key="atradeaday_run",
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    if run:
        params = ATradeADayParams(
            rr_ratio       = rr_ratio,
            risk_per_trade = risk_per_trade,
            commission_pct = commission,
        )
        try:
            with st.spinner("Running A Trade A Day strategy..."):
                results: ATradeADayResults = run_atradeaday(df.copy(), params)

            st.session_state["atradeaday_results"] = results

            r = results.backtest
            if r.num_trades == 0:
                st.warning(
                    "Strategy ran but found 0 qualifying setups. "
                    "Try loading more days of data from the sidebar."
                )
            else:
                n_pot = len(results.potential_entries)
                extra = f" · {n_pot} additional potential setups identified" if n_pot > 0 else ""
                st.success(f"✅ {r.num_trades} trades found across {n_days} days.{extra}")

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    results: ATradeADayResults = st.session_state.get("atradeaday_results")

    if results is None:
        return

    r = results.backtest

    if r.num_trades > 0:

        # Primary metrics
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Return",  f"{r.total_return_pct:.2f}%")
        c2.metric("CAGR",    f"{r.cagr:.2f}%")
        c3.metric("Sharpe",  f"{r.sharpe_ratio:.3f}")
        c4.metric("Sortino", f"{r.sortino_ratio:.3f}")
        c5.metric("Calmar",  f"{r.calmar_ratio:.3f}")
        c6.metric("Max DD",  f"{r.max_drawdown_pct:.2f}%")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades",     r.num_trades)
        c2.metric("Win %",      f"{r.win_rate:.1f}%")
        c3.metric("PF",         f"{r.profit_factor:.2f}")
        c4.metric("Expectancy", f"${r.expectancy:.2f}")
        c5.metric("Payoff",     f"{r.payoff_ratio:.2f}")
        c6.metric("Mkt Time",   f"{r.pct_time_in_market:.0f}%")

        with st.expander("📊 Detailed Metrics", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Avg Win",      f"${r.avg_winner:.2f}")
            c2.metric("Avg Loss",     f"${abs(r.avg_loser):.2f}")
            c3.metric("Avg Win %",    f"{r.avg_winner_pct:.2f}%")
            c4.metric("Avg Loss %",   f"{abs(r.avg_loser_pct):.2f}%")
            c5.metric("Max Consec L", r.max_consecutive_losses)
            c6.metric("Max Consec W", r.max_consecutive_wins)
            c1, c2 = st.columns(2)
            c1.metric("Avg Bars Held", f"{r.avg_bars_held:.1f}")
            c2.metric("Longest DD",    f"{r.longest_drawdown_bars} bars")

        st.plotly_chart(create_equity_chart(r), use_container_width=True, config=PLOTLY_CONFIG)

        st.plotly_chart(
            create_price_chart_with_trades(df, r.trades),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        # ── Trade log ─────────────────────────────────────────────────────────
        with st.expander("📋 Trade Log", expanded=False):
            trade_rows = [{
                "Date":      str(t.entry_date)[:16],
                "Direction": t.direction.upper(),
                "Entry":     f"{t.entry_price:.4f}",
                "Exit":      f"{t.exit_price:.4f}" if t.exit_price else "—",
                "Reason":    t.exit_reason or "—",
                "P&L $":     f"{t.pnl:+.2f}",
                "P&L %":     f"{t.pnl_pct:+.2f}%",
                "Bars":      t.bars_held,
            } for t in r.trades]
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True)

    # ── Potential entries ─────────────────────────────────────────────────────
    if results.potential_entries:
        st.divider()
        st.markdown(f"### 🔍 Additional Potential Setups — {len(results.potential_entries)} found")
        st.caption(
            "These are valid FVG setups that appeared on days where the primary trade "
            "was already taken. They were NOT executed (one trade per day rule) but are "
            "shown here for analysis."
        )

        pot_rows = [{
            "Date":       str(p.date)[:16],
            "Direction":  p.direction.upper(),
            "Entry":      f"{p.entry_price:.4f}",
            "SL":         f"{p.sl_price:.4f}",
            "TP":         f"{p.tp_price:.4f}",
            "FVG Top":    f"{p.fvg_top:.4f}",
            "FVG Bottom": f"{p.fvg_bottom:.4f}",
            "Day High":   f"{p.day_high:.4f}",
            "Day Low":    f"{p.day_low:.4f}",
        } for p in results.potential_entries]

        st.dataframe(pd.DataFrame(pot_rows), use_container_width=True)
