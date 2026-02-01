from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PMMemo:
    ts_utc: datetime
    status: str  # Blocked / Ready / Executed / Needs Approval
    headline: str
    blockers: List[str]
    warnings: List[str]
    missing_prices: List[str]
    cash_summary: Dict[str, Any]
    turnover_summary: Dict[str, Any]
    top_turnover_contributors: List[Dict[str, Any]]
    top_drift: List[Dict[str, Any]]
    recommended_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ts_utc"] = self.ts_utc.isoformat()
        return d


def _g(obj: Any, key: str, default=None):
    """Get a field from dict or object (dataclass)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_missing_prices(plan: Any, gates: Any) -> List[str]:
    """
    Extract missing-price symbols from plan warnings.
    Works if plan is TradePlan object or dict.
    """
    missing = set()

    for w in (_g(plan, "warnings", []) or []):
        s = str(w)
        if "Missing price for" in s:
            # expected: "Missing price for XYZ; ..."
            try:
                after = s.split("Missing price for", 1)[1].strip()
                sym = after.split(";", 1)[0].strip().upper()
                if sym:
                    missing.add(sym)
            except Exception:
                pass

    return sorted(missing)


def _top_turnover_contributors(plan: Any, nav: Optional[float], k: int = 8) -> List[Dict[str, Any]]:
    """
    Best-effort contributor ranking.
    Uses only fields that actually exist. No fabricated notional.
    """
    contrib: List[Tuple[float, Dict[str, Any]]] = []

    orders = _g(plan, "orders", []) or []
    for o in orders:
        sym = str(_g(o, "symbol", "") or "").strip().upper()
        if not sym:
            continue

        notional = _g(o, "notional_est", None)
        notional = _safe_float(notional)

        if notional is None:
            # Try qty * price_hint if both exist (only if real fields exist)
            q = _safe_float(_g(o, "quantity", None))
            p = _safe_float(_g(o, "price_hint", None))
            if q is not None and p is not None:
                notional = abs(q * p)

        if notional is None:
            continue

        pct_nav = (notional / nav) if (nav is not None and nav > 0) else None

        row = {
            "symbol": sym,
            "action": _g(o, "action", None),
            "quantity": _g(o, "quantity", None),
            "notional_est": notional,
            "pct_nav": pct_nav,
            "reason": _g(o, "reason", None),
        }
        contrib.append((float(notional), row))

    contrib.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in contrib[:k]]


def _top_drift(plan: Any, k: int = 10) -> List[Dict[str, Any]]:
    cw = _g(plan, "current_weights", {}) or {}
    tw = _g(plan, "target_weights", {}) or {}

    rows: List[Dict[str, Any]] = []
    syms = set(cw.keys()) | set(tw.keys())

    for sym in syms:
        cur = _safe_float(cw.get(sym, 0.0)) or 0.0
        tgt = _safe_float(tw.get(sym, 0.0)) or 0.0
        d = abs(tgt - cur)

        rows.append(
            {
                "symbol": sym,
                "current_weight": cur,
                "target_weight": tgt,
                "abs_drift": d,
                "delta": (tgt - cur),
            }
        )

    rows.sort(key=lambda r: float(r["abs_drift"]), reverse=True)
    return rows[:k]


def build_pm_memo(
    *,
    snapshot: Any,
    signals: Any,
    targets: Any,
    plan: Any,
    gates: Any,
    auto_trade: bool,
    armed: bool,
) -> PMMemo:
    ts = datetime.utcnow()

    nav = _safe_float(_g(snapshot, "net_liquidation", None)) or _safe_float(_g(plan, "nav", None))
    cash = _safe_float(_g(snapshot, "total_cash", None))
    proj_cash_w = _safe_float(_g(plan, "projected_cash_weight", None))  # may not exist; OK

    missing = _extract_missing_prices(plan, gates)

    blockers = list(_g(gates, "block_reasons", []) or [])
    warnings = list(_g(gates, "warnings", []) or [])

    blocked = bool(_g(gates, "blocked", False))
    if blocked:
        status = "Blocked"
        headline = "Execution blocked by hard gates."
    else:
        status = "Ready (Auto)" if (auto_trade and armed) else "Ready"
        headline = "Plan is executable under current gates."

    # Recommended actions per your policy (deterministic)
    rec: List[str] = []
    if missing:
        rec.append(f"Missing prices skipped this cycle: {', '.join(missing)}. Fix market data to include them next run.")

    for r in blockers:
        if "Turnover" in r and "exceeds cap" in r:
            rec.append("Turnover exceeds cap (hard blocker). Reduce target changes or accept multi-cycle rebalancing (no auto scaling).")
        if "Projected cash" in r and "< cash_min" in r:
            if auto_trade and armed:
                rec.append("Auto+armed: execute SELL-FIRST subset (SELL orders only), then re-plan.")
            else:
                rec.append("Manual: run SELL-FIRST (SELL orders only) to restore cash, then re-plan.")

    if not rec and not blocked:
        rec.append("No blockers detected. Review proposed actions and proceed if aligned with strategy.")

    return PMMemo(
        ts_utc=ts,
        status=status,
        headline=headline,
        blockers=blockers,
        warnings=warnings,
        missing_prices=missing,
        cash_summary={
            "nav": nav,
            "cash": cash,
            "projected_cash_weight": proj_cash_w,
            "cash_target": _g(targets, "cash_target", None),
        },
        turnover_summary={
            "turnover_est": _g(plan, "turnover_est", None),
            "num_trades": _g(plan, "num_trades", None),
        },
        top_turnover_contributors=_top_turnover_contributors(plan, nav),
        top_drift=_top_drift(plan),
        recommended_actions=rec,
    )
