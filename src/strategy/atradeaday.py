"""
src/strategy/atradeaday.py
==========================
A Trade A Day — standalone backtest engine.

Strategy rules:
  1. Each day, identify the first 5-MINUTE candle (the 9:30 candle).
     If data is 1m or 2m, resample to 5m first to get the correct candle.
     If data is coarser than 5m (15m, 30m, 60m...), refuse with a clear error.
     Mark its high and low. That's the entire analysis for the day.
  2. From the NEXT candle onwards, look for a Fair Value Gap that breaks
     through day_high (bullish) or day_low (bearish).
     FVG = 3-candle pattern: c0.high < c2.low (bull) or c0.low > c2.high (bear).
     The middle candle must cross the day level.
  3. Wait for price to pull back into the FVG zone.
  4. Wait for an engulfing candle that completely covers the pullback candle.
  5. Enter at the close of the engulfing candle.
  6. SL = low/high of the first FVG candle.
  7. TP = entry +/- risk x rr_ratio (default 3:1).
  8. One trade per day maximum (the FIRST valid setup is taken).
     All additional setups found that day are recorded as potential_entries
     for extra analysis — they are NOT traded.

Returns ATradeADayResults which contains:
  - backtest  : BacktestResults (identical schema to BacktestEngine output)
  - potential_entries : List[PotentialEntry] (extra setups found but not traded)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from src.backtest import BacktestResults, Trade


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

# Intervals that are valid (≤ 5 minutes)
_VALID_INTERVALS = {'1m', '2m', '5m'}

# Intervals that need resampling to 5m before use
_RESAMPLE_INTERVALS = {'1m', '2m'}


@dataclass
class ATradeADayParams:
    rr_ratio: float = 3.0
    risk_per_trade: float = 100.0
    commission_pct: float = 0.05


@dataclass
class PotentialEntry:
    """
    A valid FVG setup found on a given day that was NOT traded
    because the primary trade for that day already occurred.
    Used for extra analysis display in the UI.
    """
    date: pd.Timestamp
    direction: str          # 'long' or 'short'
    fvg_top: float
    fvg_bottom: float
    sl_price: float
    entry_price: float      # close of the engulfing candle
    tp_price: float
    day_high: float         # the reference 5-min candle high
    day_low: float          # the reference 5-min candle low


@dataclass
class ATradeADayResults:
    """
    Full output of run_atradeaday().
    backtest          → standard BacktestResults for metrics display
    potential_entries → extra setups found on days where a trade was already taken
    """
    backtest: BacktestResults
    potential_entries: List[PotentialEntry] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Interval validation and 5-min candle extraction
# ─────────────────────────────────────────────────────────────────────────────

def validate_interval(df: pd.DataFrame) -> str:
    """
    Read the interval from df.attrs and validate it is ≤ 5 minutes.

    Returns the interval string ('1m', '2m', '5m').
    Raises ValueError with a user-friendly message if interval is too coarse.
    """
    interval = df.attrs.get('interval', None)

    if interval is None:
        # Try to infer from median delta
        if len(df) >= 2:
            deltas  = pd.Series(df.index).diff().dropna()
            seconds = deltas.median().total_seconds()
            if seconds <= 60:   interval = '1m'
            elif seconds <= 120: interval = '2m'
            elif seconds <= 300: interval = '5m'
            else:
                raise ValueError(
                    f"Could not detect interval and data appears coarser than 5 minutes. "
                    f"A Trade A Day requires 1m, 2m, or 5m data. "
                    f"Please reload data from the sidebar with a ≤5m interval."
                )
        else:
            raise ValueError("Not enough data to detect interval.")

    if interval not in _VALID_INTERVALS:
        raise ValueError(
            f"Loaded data is '{interval}' — too coarse for A Trade A Day. "
            f"This strategy requires 1m, 2m, or 5m data so the first 5-minute "
            f"candle can be identified correctly. "
            f"Please reload from the sidebar with interval = 1m, 2m, or 5m."
        )

    return interval


def _get_first_5min_candle(day_df: pd.DataFrame, interval: str) -> Optional[pd.Series]:
    """
    Extract the first 5-minute candle of the day.

    - 5m  → first bar of the day directly
    - 1m / 2m → resample the full day to 5m, return the first 5-min bar.
                 This correctly aggregates open/high/low/close/volume across
                 the sub-minute bars that make up the first 5 minutes.

    Returns None if the day has no usable bars after resampling.
    """
    if interval == '5m':
        return day_df.iloc[0] if not day_df.empty else None

    # Resample to 5-minute bars
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in day_df.columns:
        agg['volume'] = 'sum'

    resampled = day_df.resample('5min').agg(agg).dropna(subset=['open', 'close'])

    if resampled.empty:
        return None

    return resampled.iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Pattern detection
# ─────────────────────────────────────────────────────────────────────────────

def _find_all_fvg_breaks(
    bars: pd.DataFrame,
    day_high: float,
    day_low: float,
) -> List[Tuple[str, int, float, float, float]]:
    """
    Scan ALL 3-candle windows in `bars` for Fair Value Gaps that confirm
    a break of day_high or day_low.

    Returns a list of (direction, c2_index, fvg_top, fvg_bottom, sl_price).
    The list is ordered by appearance — first item is the primary setup,
    subsequent items are potential extras.
    """
    results = []

    for i in range(2, len(bars)):
        c0 = bars.iloc[i - 2]
        c1 = bars.iloc[i - 1]
        c2 = bars.iloc[i]

        # ── Bullish FVG ───────────────────────────────────────────────────────
        if c2['low'] > c0['high']:
            if c1['high'] >= day_high or c2['close'] > day_high or c1['close'] > day_high:
                results.append((
                    'long', i,
                    c2['low'],   # fvg_top
                    c0['high'],  # fvg_bottom
                    c0['low'],   # sl_price
                ))

        # ── Bearish FVG ───────────────────────────────────────────────────────
        elif c2['high'] < c0['low']:
            if c1['low'] <= day_low or c2['close'] < day_low or c1['close'] < day_low:
                results.append((
                    'short', i,
                    c0['low'],   # fvg_top
                    c2['high'],  # fvg_bottom
                    c0['high'],  # sl_price
                ))

    return results


def _find_engulfing_entry(
    bars: pd.DataFrame,
    direction: str,
    fvg_top: float,
    fvg_bottom: float,
) -> Optional[Tuple[int, pd.Series, pd.Series]]:
    """
    Scan `bars` for pullback into FVG zone then an engulfing candle.

    Relaxed: engulf covers the body (open-close) of the pullback candle,
    not necessarily the full wicks — catches more real-world setups.

    Returns (bar_position, pullback_candle, engulf_candle) or None.
    """
    in_pullback  = False
    pullback_bar = None

    for i in range(len(bars)):
        bar = bars.iloc[i]

        if not in_pullback:
            if direction == 'long':
                if bar['low'] <= fvg_top and bar['close'] >= fvg_bottom:
                    in_pullback  = True
                    pullback_bar = bar
            else:
                if bar['high'] >= fvg_bottom and bar['close'] <= fvg_top:
                    in_pullback  = True
                    pullback_bar = bar
        else:
            pb_body_high = max(pullback_bar['open'], pullback_bar['close'])
            pb_body_low  = min(pullback_bar['open'], pullback_bar['close'])

            if direction == 'long':
                if (bar['close'] > pb_body_high and bar['open'] < pb_body_low) or \
                   (bar['close'] > pullback_bar['high'] and bar['open'] < pb_body_high):
                    return (i, pullback_bar, bar)
                if bar['low'] > fvg_top * 1.002:
                    in_pullback  = False
                    pullback_bar = None
            else:
                if (bar['close'] < pb_body_low and bar['open'] > pb_body_high) or \
                   (bar['close'] < pullback_bar['low'] and bar['open'] > pb_body_low):
                    return (i, pullback_bar, bar)
                if bar['high'] < fvg_bottom * 0.998:
                    in_pullback  = False
                    pullback_bar = None

    return None


def _simulate_exit(
    bars: pd.DataFrame,
    direction: str,
    sl_price: float,
    tp_price: float,
) -> Tuple[float, str, int]:
    """
    Walk through remaining bars to find SL or TP hit.
    Gap-aware: if bar opens past the level, fill at open.
    Returns (exit_price, exit_reason, bars_held).
    """
    for i, (_, bar) in enumerate(bars.iterrows()):
        sl_hit = tp_hit = False
        sl_fill = tp_fill = None

        if direction == 'long':
            if bar['low'] <= sl_price:
                sl_hit  = True
                sl_fill = min(sl_price, bar['open'])
            if bar['high'] >= tp_price:
                tp_hit  = True
                tp_fill = max(tp_price, bar['open'])
        else:
            if bar['high'] >= sl_price:
                sl_hit  = True
                sl_fill = max(sl_price, bar['open'])
            if bar['low'] <= tp_price:
                tp_hit  = True
                tp_fill = min(tp_price, bar['open'])

        if sl_hit and tp_hit:
            sl_dist = abs(bar['open'] - sl_fill)
            tp_dist = abs(bar['open'] - tp_fill)
            if tp_dist <= sl_dist:
                return tp_fill, 'take_profit', i + 1
            else:
                return sl_fill, 'stop_loss', i + 1
        elif tp_hit:
            return tp_fill, 'take_profit', i + 1
        elif sl_hit:
            return sl_fill, 'stop_loss', i + 1

    last_close = bars.iloc[-1]['close'] if len(bars) > 0 else sl_price
    return last_close, 'end_of_data', len(bars)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (mirrors BacktestEngine._calculate_metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_bars_per_year(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 252
    deltas  = pd.Series(df.index).diff().dropna()
    seconds = deltas.median().total_seconds()
    if seconds <= 120:      return 252 * 390
    elif seconds <= 600:    return 252 * 78
    elif seconds <= 1800:   return 252 * 26
    elif seconds <= 3600:   return 252 * 13
    elif seconds <= 7200:   return 252 * 7
    elif seconds <= 172800: return 252
    elif seconds <= 864000: return 52
    else:                   return 12


def _calculate_metrics(
    trades: List[Trade],
    equity_curve: pd.Series,
    realized_equity: pd.Series,
    initial_capital: float,
    commission_pct: float,
    bars_per_year: int,
) -> BacktestResults:

    if not trades:
        return BacktestResults(
            trades=[],
            equity_curve=equity_curve,
            realized_equity=realized_equity,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            bars_per_year=bars_per_year,
        )

    winners = [t for t in trades if t.pnl > 0]
    losers  = [t for t in trades if t.pnl <= 0]

    total_return     = equity_curve.iloc[-1] - initial_capital
    total_return_pct = total_return / initial_capital * 100

    n_bars = len(equity_curve)
    cagr   = 0.0
    if n_bars > 1 and equity_curve.iloc[-1] > 0:
        cagr = ((equity_curve.iloc[-1] / initial_capital) ** (bars_per_year / n_bars) - 1) * 100

    num_trades    = len(trades)
    win_rate      = len(winners) / num_trades * 100
    gross_profit  = sum(t.pnl for t in winners) if winners else 0.0
    gross_loss    = abs(sum(t.pnl for t in losers)) if losers else 0.0
    profit_factor = min(gross_profit / gross_loss, 999.99) if gross_loss > 0 else (999.99 if gross_profit > 0 else 0.0)

    avg_winner     = float(np.mean([t.pnl     for t in winners])) if winners else 0.0
    avg_loser      = float(np.mean([t.pnl     for t in losers]))  if losers  else 0.0
    avg_winner_pct = float(np.mean([t.pnl_pct for t in winners])) if winners else 0.0
    avg_loser_pct  = float(np.mean([t.pnl_pct for t in losers]))  if losers  else 0.0
    avg_trade      = float(np.mean([t.pnl     for t in trades]))
    avg_bars       = float(np.mean([t.bars_held for t in trades]))

    wr_frac    = len(winners) / num_trades
    expectancy = wr_frac * avg_winner - (1.0 - wr_frac) * abs(avg_loser)
    payoff     = min(avg_winner / abs(avg_loser), 999.99) if avg_loser != 0 else 999.99

    peak       = equity_curve.expanding().max()
    drawdown   = equity_curve - peak
    max_dd     = float(drawdown.min())
    max_dd_pct = float((drawdown / peak).min() * 100) if peak.max() > 0 else 0.0

    in_dd      = drawdown < 0
    dd_groups  = (~in_dd).cumsum()
    longest_dd = int(in_dd.groupby(dd_groups).sum().max()) if in_dd.any() else 0

    returns        = equity_curve.pct_change().dropna()
    active_returns = returns[returns != 0]
    n_total        = len(returns)
    n_active       = len(active_returns)

    active_bpy = n_active * (bars_per_year / n_total) if n_total > 0 and n_active > 0 else bars_per_year

    sharpe  = float((active_returns.mean() / active_returns.std()) * np.sqrt(active_bpy)) \
              if n_active > 1 and active_returns.std() > 0 else 0.0

    neg_active = active_returns[active_returns < 0]
    sortino    = float((active_returns.mean() / neg_active.std()) * np.sqrt(active_bpy)) \
                 if len(neg_active) > 1 and neg_active.std() > 0 else sharpe

    calmar = abs(cagr / max_dd_pct) if max_dd_pct != 0 else 0.0

    max_cl = max_cw = cur_l = cur_w = 0
    for t in trades:
        if t.pnl <= 0:
            cur_l += 1; cur_w = 0; max_cl = max(max_cl, cur_l)
        else:
            cur_w += 1; cur_l = 0; max_cw = max(max_cw, cur_w)

    pct_in_market = (sum(t.bars_held for t in trades) / n_bars * 100) if n_bars > 0 else 0.0

    return BacktestResults(
        trades=trades,
        equity_curve=equity_curve,
        realized_equity=realized_equity,
        total_return=total_return,
        total_return_pct=total_return_pct,
        cagr=cagr,
        num_trades=num_trades,
        winners=len(winners),
        losers=len(losers),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        payoff_ratio=payoff,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
        avg_winner_pct=avg_winner_pct,
        avg_loser_pct=avg_loser_pct,
        avg_trade=avg_trade,
        avg_bars_held=avg_bars,
        max_consecutive_losses=max_cl,
        max_consecutive_wins=max_cw,
        longest_drawdown_bars=longest_dd,
        pct_time_in_market=pct_in_market,
        avg_mae=0.0,
        avg_mfe=0.0,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        bars_per_year=bars_per_year,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_atradeaday(df: pd.DataFrame, params: ATradeADayParams) -> ATradeADayResults:
    """
    Run the A Trade A Day strategy.

    Requires 1m, 2m, or 5m data loaded in the sidebar.
    Raises ValueError (caught by the UI) if the interval is too coarse.

    Returns ATradeADayResults:
      .backtest          → full BacktestResults for metrics/charts
      .potential_entries → extra setups found on days where the primary
                           trade was already taken (for analysis display)
    """
    if df is None or df.empty:
        empty = pd.Series(dtype=float)
        empty_bt = BacktestResults(trades=[], equity_curve=empty, realized_equity=empty)
        return ATradeADayResults(backtest=empty_bt)

    # ── Validate interval ─────────────────────────────────────────────────────
    interval = validate_interval(df)   # raises ValueError if coarse

    df = df.copy()
    bars_per_year   = _estimate_bars_per_year(df)
    initial_capital = params.risk_per_trade * 50
    capital         = float(initial_capital)

    trades: List[Trade]          = []
    potentials: List[PotentialEntry] = []

    equity_index = df.index.tolist()
    running_eq   = {ts: initial_capital for ts in equity_index}

    df['_date'] = df.index.normalize()

    for date, day_df in df.groupby('_date'):
        if len(day_df) < 4:
            continue

        # ── Step 1: Get the first 5-min candle of the day ─────────────────────
        opening_bar = _get_first_5min_candle(day_df, interval)
        if opening_bar is None:
            continue

        day_high = float(opening_bar['high'])
        day_low  = float(opening_bar['low'])

        # All bars AFTER the opening candle, same day
        post_open = day_df[day_df.index > opening_bar.name] \
                    if hasattr(opening_bar, 'name') and opening_bar.name in day_df.index \
                    else day_df.iloc[1:]

        if len(post_open) < 3:
            continue

        # ── Step 2: Find ALL FVG breaks on this day ───────────────────────────
        all_fvgs = _find_all_fvg_breaks(post_open, day_high, day_low)
        if not all_fvgs:
            continue

        trade_taken = False

        for fvg_idx, fvg_result in enumerate(all_fvgs):
            direction, fvg_c2_pos, fvg_top, fvg_bottom, sl_price_raw = fvg_result

            post_fvg = post_open.iloc[fvg_c2_pos + 1:]
            if len(post_fvg) < 1:
                continue

            entry_result = _find_engulfing_entry(post_fvg, direction, fvg_top, fvg_bottom)
            if entry_result is None:
                continue

            _, _, engulf_bar = entry_result
            entry_price = float(engulf_bar['close'])
            sl_price    = float(sl_price_raw)

            if direction == 'long':
                risk_per_unit = entry_price - sl_price
                tp_price      = entry_price + risk_per_unit * params.rr_ratio
            else:
                risk_per_unit = sl_price - entry_price
                tp_price      = entry_price - risk_per_unit * params.rr_ratio

            if risk_per_unit <= 0 or risk_per_unit > entry_price * 0.15:
                continue

            # ── First valid setup of the day → TRADE IT ───────────────────────
            if not trade_taken:
                trade_taken = True

                position_size_dollars = params.risk_per_trade / (risk_per_unit / entry_price)
                remaining = post_fvg[post_fvg.index > engulf_bar.name]

                if len(remaining) == 0:
                    exit_price  = entry_price
                    exit_reason = 'end_of_data'
                    bars_held   = 0
                    exit_ts     = engulf_bar.name
                else:
                    exit_price, exit_reason, bars_held = _simulate_exit(
                        remaining, direction, sl_price, tp_price
                    )
                    exit_idx = min(bars_held - 1, len(remaining) - 1)
                    exit_ts  = remaining.index[exit_idx]

                if direction == 'long':
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                pnl_pct -= params.commission_pct * 2
                pnl      = position_size_dollars * (pnl_pct / 100)
                capital += pnl

                try:
                    entry_iloc = equity_index.index(engulf_bar.name)
                    exit_iloc  = equity_index.index(exit_ts)
                except ValueError:
                    entry_iloc = exit_iloc = 0

                trades.append(Trade(
                    entry_idx    = entry_iloc,
                    entry_date   = engulf_bar.name,
                    entry_price  = entry_price,
                    direction    = direction,
                    size_dollars = position_size_dollars,
                    exit_idx     = exit_iloc,
                    exit_date    = exit_ts,
                    exit_price   = exit_price,
                    exit_reason  = exit_reason,
                    pnl          = pnl,
                    pnl_pct      = pnl_pct,
                    bars_held    = bars_held,
                    mae          = 0.0,
                    mfe          = 0.0,
                ))

                # Update equity from exit bar onwards
                updating = False
                for ts in equity_index:
                    if ts == exit_ts:
                        updating = True
                    if updating:
                        running_eq[ts] = capital

            else:
                # ── Subsequent setups on same day → POTENTIAL only ─────────────
                potentials.append(PotentialEntry(
                    date        = engulf_bar.name,
                    direction   = direction,
                    fvg_top     = fvg_top,
                    fvg_bottom  = fvg_bottom,
                    sl_price    = sl_price,
                    entry_price = entry_price,
                    tp_price    = tp_price,
                    day_high    = day_high,
                    day_low     = day_low,
                ))

    equity_curve    = pd.Series(running_eq, index=equity_index)
    realized_equity = equity_curve.copy()

    backtest = _calculate_metrics(
        trades, equity_curve, realized_equity,
        initial_capital, params.commission_pct, bars_per_year,
    )

    return ATradeADayResults(backtest=backtest, potential_entries=potentials)
