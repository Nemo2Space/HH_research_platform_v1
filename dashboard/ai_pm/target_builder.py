from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import DEFAULT_CONSTRAINTS, STRATEGIES, RiskConstraints, StrategyProfile
from .models import SignalSnapshot, TargetWeights


def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    s = sum(v for v in raw.values() if v and v > 0)
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in raw.items() if v and v > 0}


def _norm0100_to_01(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        v = float(x)
        # If looks like 0..100 score, map to 0..1. If already 0..1, keep.
        if v > 1.0:
            return max(0.0, min(1.0, v / 100.0))
        return max(0.0, min(1.0, v))
    except Exception:
        return None


def _extract_probability(sym: str, raw: Dict) -> Optional[float]:
    """
    Probability in [0..1].
    Accepts canonical keys and merged-suffix variants. Never fabricates.
    """
    candidates = [
        raw.get("ai_probability"),
        raw.get("ai_probability_score"),
        raw.get("ai_probability_signal"),
        raw.get("ai_prob"),
        raw.get("AI Prob"),
        raw.get("probability"),
        raw.get("probability_score"),
        raw.get("probability_signal"),
        raw.get("win_probability"),
        raw.get("prob_win"),
        raw.get("prob_win_5d"),
        raw.get("predicted_probability"),
        raw.get("likelihood_score"),
        raw.get("likelihood_score_score"),
        raw.get("likelihood_score_signal"),
        raw.get("Likelihood"),
        raw.get("likelihood_score_norm"),
    ]
    for c in candidates:
        p = _norm0100_to_01(c)
        if p is not None:
            return p
    return None


def _extract_ev(raw: Dict) -> Tuple[Optional[float], Optional[str]]:
    """
    Expected value if present.
    Returns: (ev_decimal, note)
    - Never fabricates; returns None if missing.
    """
    candidates = [
        raw.get("ai_ev"),
        raw.get("expected_value"),
        raw.get("ev"),
        raw.get("expected_value_5d"),
    ]
    for c in candidates:
        try:
            if c is None or (isinstance(c, float) and pd.isna(c)):
                continue
            v = float(c)
            # If EV appears in percent units (e.g., 1.5 == 1.5%), normalize to decimal.
            if abs(v) > 1.0:
                return v / 100.0, "EV appeared percent-like; normalized by /100."
            return v, None
        except Exception:
            continue
    return None, None


def _pick_top_symbols_by_score(
        signals: SignalSnapshot,
        strategy: StrategyProfile,
        max_holdings: int,
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Blended ranking for core strategies.
    Uses normalized fields from SignalRow.

    NO FALLBACKS: Returns empty list with notes if no data matches.
    """
    ranked: List[Tuple[str, float]] = []
    notes: List[str] = []

    total_rows = 0
    rows_with_scores = 0
    rows_missing_scores = 0

    for sym, row in (signals.rows or {}).items():
        total_rows += 1
        ers = row.expected_return_score
        conv = row.conviction_score
        raw = row.raw or {}

        total_score = raw.get("total_score")
        signal_strength = raw.get("signal_strength")
        likelihood = raw.get("likelihood_score")

        def _norm0100(x):
            try:
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return None
                return float(x) / 100.0
            except Exception:
                return None

        total_n = _norm0100(total_score)
        strength_n = _norm0100(signal_strength)
        like_n = _norm0100(likelihood)

        parts: List[Tuple[float, Optional[float]]] = []

        if getattr(strategy, "w_total_score", 0) > 0:
            parts.append((strategy.w_total_score, total_n if total_n is not None else ers))
        if getattr(strategy, "w_signal_strength", 0) > 0:
            parts.append((strategy.w_signal_strength, strength_n if strength_n is not None else ers))
        if getattr(strategy, "w_committee_conviction", 0) > 0:
            parts.append((strategy.w_committee_conviction, like_n if like_n is not None else conv))

        # If strategy weights are all zero, use total_score directly
        if not parts:
            if total_n is not None:
                ranked.append((sym, float(total_n)))
                rows_with_scores += 1
            elif ers is not None:
                ranked.append((sym, float(ers)))
                rows_with_scores += 1
            else:
                rows_missing_scores += 1
            continue

        num = 0.0
        den = 0.0
        for w, v in parts:
            if v is None:
                continue
            num += float(w) * float(v)
            den += float(w)

        if den <= 0:
            rows_missing_scores += 1
            continue

        score = num / den
        if score <= 0.0:
            rows_missing_scores += 1
            continue

        ranked.append((sym, score))
        rows_with_scores += 1

    notes.append(f"Signals scanned: {total_rows}")
    notes.append(f"Rows with valid scores: {rows_with_scores}")
    if rows_missing_scores > 0:
        notes.append(f"Rows missing required scores: {rows_missing_scores}")

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[: int(max_holdings)], notes


def _pick_top_symbols_ai_probability(
        signals: SignalSnapshot,
        *,
        max_holdings: int,
        min_probability: float,
        min_ev: float,
        conservative: bool,
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    AI Probability / AI Conservative selector.

    NO FALLBACKS VERSION:
    - If not enough stocks meet criteria, returns what's available with clear warnings
    - Does NOT auto-lower thresholds or use fallback data
    """
    notes: List[str] = []

    total_rows = 0
    prob_from_raw = 0
    prob_from_conv = 0
    prob_missing = 0
    ev_missing_all = True
    ev_norm_notes = 0
    below_min_prob = 0
    below_min_ev = 0

    # EV presence scan
    for _, row in (signals.rows or {}).items():
        raw = row.raw or {}
        ev, _ = _extract_ev(raw)
        if ev is not None:
            ev_missing_all = False
            break

    if ev_missing_all:
        notes.append("⚠️ EV field not found in signals; EV filter disabled.")

    candidates: List[Tuple[str, float, Optional[float], Optional[float], str]] = []
    p_values: List[float] = []

    for sym, row in (signals.rows or {}).items():
        total_rows += 1
        raw = row.raw or {}

        p = _extract_probability(sym, raw)
        src = "raw"
        if p is not None:
            prob_from_raw += 1

        if p is None:
            # Try conviction_score as source
            try:
                cs = getattr(row, "conviction_score", None)
                if cs is not None:
                    p = float(cs)
                    src = "conviction_score"
                    prob_from_conv += 1
            except Exception:
                p = None

        if p is None:
            prob_missing += 1
            continue

        p_values.append(p)

        if p < float(min_probability):
            below_min_prob += 1
            continue

        ev, ev_note = _extract_ev(raw)
        if ev_note:
            ev_norm_notes += 1

        # Enforce min_ev strictly if EV exists somewhere
        if (not ev_missing_all) and (ev is None or ev < float(min_ev)):
            below_min_ev += 1
            continue

        rf = None
        try:
            rf0 = getattr(row, "risk_flag_score", None)
            rf = float(rf0) if rf0 is not None else None
        except Exception:
            rf = None

        score = float(p)

        if (not ev_missing_all) and (ev is not None):
            score = score * max(float(ev), 0.0)

        if conservative and rf is not None:
            try:
                rf = max(0.0, min(1.0, float(rf)))
                score = score * (1.0 - rf)
            except Exception:
                pass

        if score <= 0.0:
            continue

        sym_u = str(sym).strip().upper()
        candidates.append((sym_u, float(p), ev, rf, src))

    # Build detailed notes
    notes.append(f"Signals rows scanned: {total_rows}")
    notes.append(f"Probability sources: raw={prob_from_raw}, conviction_score={prob_from_conv}, missing={prob_missing}")

    if p_values:
        notes.append(
            f"Probability range: min={min(p_values):.3f}, max={max(p_values):.3f}, median={sorted(p_values)[len(p_values) // 2]:.3f}")

    if below_min_prob > 0:
        notes.append(f"⚠️ {below_min_prob} stocks below min_probability={min_probability:.2f}")

    if below_min_ev > 0:
        notes.append(f"⚠️ {below_min_ev} stocks below min_ev={min_ev:.4f}")

    # NO FALLBACK: If nothing passed, return empty with clear warning
    if not candidates:
        notes.append(f"❌ NO STOCKS PASSED STRATEGY CRITERIA")
        notes.append(f"   Thresholds: min_probability={min_probability:.2f}" + (
            f", min_ev={min_ev:.4f}" if not ev_missing_all else ""))
        notes.append("   💡 Options: (1) Lower thresholds, (2) Run scanner on more stocks, (3) Try different strategy")
        return [], notes

    # Rank by score
    ranked_scored: List[Tuple[str, float]] = []
    for sym, p, ev, rf, src in candidates:
        score = float(p)
        if (not ev_missing_all) and (ev is not None):
            score = score * max(float(ev), 0.0)
        if conservative and rf is not None:
            try:
                rf = max(0.0, min(1.0, float(rf)))
                score = score * (1.0 - rf)
            except Exception:
                pass
        if score > 0.0:
            ranked_scored.append((sym, score))

    ranked_scored.sort(key=lambda x: x[1], reverse=True)
    ranked_scored = ranked_scored[: int(max_holdings)]

    notes.append(f"✓ {len(ranked_scored)} stocks matched criteria (max_holdings={max_holdings})")

    return ranked_scored, notes


def build_target_weights(
        *,
        signals: SignalSnapshot,
        strategy_key: str,
        constraints: RiskConstraints = DEFAULT_CONSTRAINTS,
        max_holdings_override: Optional[int] = None,
) -> TargetWeights:
    """
    Target builder with NO FALLBACKS and configurable max_holdings.

    Changes from original:
    - Added max_holdings_override parameter for UI control
    - Returns empty weights with clear warnings if insufficient data
    - Never uses fallback/default data
    - Always provides actionable debug notes
    """
    ts = datetime.utcnow()
    strategy = STRATEGIES.get(strategy_key) or STRATEGIES.get("core_passive_drift")

    if strategy and getattr(strategy, "constraints_override", None):
        constraints = strategy.constraints_override

    # Apply max_holdings override if provided by user
    max_holdings = max_holdings_override if max_holdings_override else getattr(strategy, "max_holdings", 25)
    min_weight = getattr(strategy, "min_weight", 0.01) if strategy else 0.01

    notes: List[str] = []
    notes.append(f"Strategy: {strategy_key}")
    notes.append(f"Max holdings: {max_holdings}")

    # Check if we have any signals at all
    if not signals or not signals.rows:
        return TargetWeights(
            ts_utc=ts,
            strategy_key=strategy_key,
            weights={},
            cash_target=constraints.cash_target,
            notes=["❌ NO SIGNALS DATA AVAILABLE", "   Run the scanner/analyzer first to generate signals for stocks."],
        )

    notes.append(f"Available signals: {len(signals.rows)} symbols")

    # ---------- AI Probability strategies ----------
    if strategy_key in ("ai_probability", "ai_conservative"):
        if strategy_key == "ai_probability":
            min_p = 0.40
            min_ev = 0.001
            conservative = False
        else:
            min_p = 0.55
            min_ev = 0.005
            conservative = True

        ranked, ai_notes = _pick_top_symbols_ai_probability(
            signals,
            max_holdings=int(max_holdings),
            min_probability=float(min_p),
            min_ev=float(min_ev),
            conservative=bool(conservative),
        )
        notes.extend(ai_notes)

        if not ranked:
            return TargetWeights(
                ts_utc=ts,
                strategy_key=strategy_key,
                weights={},
                cash_target=constraints.cash_target,
                notes=notes,
            )

        # Check minimum required stocks
        if len(ranked) < 5:
            notes.append(f"⚠️ WARNING: Only {len(ranked)} stocks matched (below recommended minimum of 5)")

        raw_w = {sym: max(float(score), 0.0) for sym, score in ranked}
        w = _normalize_weights(raw_w)

        w = {sym: wt for sym, wt in w.items() if wt >= float(min_weight or 0.0)}
        w = _normalize_weights(w)

        notes.append(f"Final portfolio: {len(w)} holdings after min_weight={min_weight:.2%} filter")
        notes.append(f"Cash target: {constraints.cash_target:.1%}")

        return TargetWeights(
            ts_utc=ts,
            strategy_key=strategy_key,
            weights=w,
            cash_target=constraints.cash_target,
            notes=notes,
        )

    # ---------- Existing blended-score strategies ----------
    if not strategy:
        return TargetWeights(
            ts_utc=ts,
            strategy_key=strategy_key,
            weights={},
            cash_target=constraints.cash_target,
            notes=[f"❌ Strategy not found: {strategy_key}", f"   Available strategies: {list(STRATEGIES.keys())}"],
        )

    ranked, score_notes = _pick_top_symbols_by_score(signals, strategy, max_holdings)
    notes.extend(score_notes)

    if not ranked:
        notes.append("❌ NO STOCKS MATCHED STRATEGY CRITERIA")
        notes.append("   Check that signals have required fields: total_score, signal_strength, likelihood_score")
        return TargetWeights(
            ts_utc=ts,
            strategy_key=strategy.key,
            weights={},
            cash_target=constraints.cash_target,
            notes=notes,
        )

    # Check minimum required stocks
    if len(ranked) < 5:
        notes.append(f"⚠️ WARNING: Only {len(ranked)} stocks matched (below recommended minimum of 5)")

    raw_w = {sym: max(float(score), 0.0) for sym, score in ranked}
    w = _normalize_weights(raw_w)

    w = {sym: wt for sym, wt in w.items() if wt >= float(strategy.min_weight or 0.0)}
    w = _normalize_weights(w)

    notes.append(f"Final portfolio: {len(w)} holdings after min_weight filter")
    notes.append(f"Cash target: {constraints.cash_target:.1%}")

    return TargetWeights(
        ts_utc=ts,
        strategy_key=strategy.key,
        weights=w,
        cash_target=constraints.cash_target,
        notes=notes,
    )