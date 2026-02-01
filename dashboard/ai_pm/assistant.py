# dashboard/ai_pm/assistant.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re
import json

from .pm_memo import build_pm_memo


@dataclass
class AssistantAnswer:
    ok: bool = True
    answer: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "answer": self.answer, "evidence": self.evidence}


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
        try:
            return _to_jsonable(x.to_dict())
        except Exception:
            pass
    if hasattr(x, "__dict__"):
        try:
            return _to_jsonable(dict(x.__dict__))
        except Exception:
            pass
    try:
        return str(x)
    except Exception:
        return None


def _safe_json(x: Any) -> str:
    try:
        return json.dumps(_to_jsonable(x), ensure_ascii=False, indent=2)
    except Exception:
        return "{}"


def _extract_ticker_from_question(question: str, snapshot: Any) -> Optional[str]:
    """
    Prefer tickers that exist in current portfolio.
    """
    q = (question or "").upper()
    candidates = re.findall(r"\b[A-Z]{1,7}\b", q)

    # build a set of portfolio symbols
    syms = set()
    try:
        pos = getattr(snapshot, "positions", None) or []
        for p in pos:
            s = getattr(p, "symbol", None)
            if s:
                syms.add(str(s).upper())
    except Exception:
        pass

    for c in candidates:
        if c in syms:
            return c

    # fallback: first candidate
    return candidates[0] if candidates else None


def _get_targets_weight(targets: Any, sym: str) -> Optional[float]:
    if not targets or not sym:
        return None
    # TargetWeights model: usually has .weights dict
    try:
        w = getattr(targets, "weights", None)
        if isinstance(w, dict):
            v = w.get(sym)
            return float(v) if v is not None else None
    except Exception:
        pass
    # dict fallback
    t = _to_jsonable(targets)
    if isinstance(t, dict):
        w = t.get("weights")
        if isinstance(w, dict) and sym in w:
            try:
                return float(w.get(sym))
            except Exception:
                return None
    return None


def _get_signal_snapshot(signals: Any, sym: str) -> Any:
    if not signals or not sym:
        return None
    # signals could be SignalSnapshot object with .by_symbol dict OR plain dict
    try:
        by = getattr(signals, "by_symbol", None)
        if isinstance(by, dict):
            return by.get(sym)
    except Exception:
        pass
    if isinstance(signals, dict):
        return signals.get(sym)
    s = _to_jsonable(signals)
    if isinstance(s, dict):
        return s.get(sym)
    return None


def _compute_portfolio_weights_from_snapshot(snapshot: Any) -> Dict[str, float]:
    """
    Uses market_value if present. If not, returns empty dict (no fake data).
    """
    weights: Dict[str, float] = {}
    try:
        pos = getattr(snapshot, "positions", None) or []
        mvs = []
        for p in pos:
            mv = getattr(p, "market_value", None)
            sym = getattr(p, "symbol", None)
            if sym and (mv is not None):
                try:
                    mvs.append((str(sym).upper(), float(mv)))
                except Exception:
                    pass
        total = sum(abs(v) for _, v in mvs)
        if total <= 0:
            return {}
        for sym, mv in mvs:
            weights[sym] = float(mv) / float(total)
    except Exception:
        return {}
    return weights


def _try_research_agent(prompt: str, history: Optional[List[Dict[str, Any]]]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Calls your existing research agent (AlphaChat wrapper).
    """
    try:
        # project path
        from src.ai.ai_research_agent_v4 import research_sync  # type: ignore
    except Exception:
        try:
            # local fallback name if you placed it differently
            from ai_research_agent_v4 import research_sync  # type: ignore
        except Exception as e:
            return False, "", {"research_error": f"research_sync import failed: {e}"}

    try:
        resp = research_sync(question=prompt, history=history or [])
        ok = bool(getattr(resp, "success", False))
        txt = str(getattr(resp, "analysis", "") or "").strip()
        ev = {"metadata": _to_jsonable(getattr(resp, "metadata", {}))}
        return ok, txt, ev
    except Exception as e:
        return False, "", {"research_error": str(e)}


def answer_pm_question(
    question: str,
    *,
    snapshot: Any = None,
    signals: Any = None,
    targets: Any = None,
    plan: Any = None,
    gates: Any = None,
    auto_trade: bool = False,
    armed: bool = False,
    strategy_key: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    enable_research: bool = True,
    debug: bool = False,
    **_ignored: Any,
) -> AssistantAnswer:
    if snapshot is None:
        return AssistantAnswer(ok=False, answer="No snapshot provided to AI assistant.", evidence={})

    q = (question or "").strip()
    ql = q.lower()

    memo = build_pm_memo(
        snapshot=snapshot,
        signals=signals,
        targets=targets,
        plan=plan,
        gates=gates,
        auto_trade=auto_trade,
        armed=armed,
    )

    # Diagnostics: valuation coverage (helps confirm "after hours" vs "snapshot discarding values")
    weights = _compute_portfolio_weights_from_snapshot(snapshot)
    missing_mv = []
    try:
        for p in (getattr(snapshot, "positions", None) or []):
            sym = str(getattr(p, "symbol", "") or "").upper()
            if not sym:
                continue
            mp = getattr(p, "market_price", None)
            mv = getattr(p, "market_value", None)
            if mp is None and mv is None:
                missing_mv.append(sym)
    except Exception:
        pass

    # --- operational: blocked / gates
    if ("blocked" in ql) or ("hard gate" in ql) or ("hard gates" in ql) or ("execution" in ql and "block" in ql):
        lines = []
        if memo.blockers:
            lines.append("Execution is blocked by hard gates for this cycle.")
            lines.append("Block reasons:")
            for b in memo.blockers:
                lines.append(f"- {b}")
        else:
            lines.append("Execution is not blocked by hard gates for this cycle.")

        if debug:
            return AssistantAnswer(
                ok=True,
                answer="\n".join(lines),
                evidence={
                    "strategy_key": strategy_key,
                    "missing_price_or_value_symbols": sorted(set(missing_mv))[:100],
                    "have_mv_weights": bool(weights),
                    "memo": memo.to_dict(),
                },
            )
        return AssistantAnswer(ok=True, answer="\n".join(lines), evidence={})

    # --- “why do I have <ticker>”
    if ("why" in ql and ("have" in ql or "in my portfolio" in ql or "hold" in ql)):
        sym = _extract_ticker_from_question(q, snapshot)
        if not sym:
            return AssistantAnswer(ok=True, answer="Please include the ticker symbol in the question (e.g., SPSK).", evidence={})

        current_w = None
        # prefer memo.top_drift if it includes current_weight
        try:
            for row in (memo.top_drift or []):
                if str(row.get("symbol", "")).upper() == sym:
                    current_w = row.get("current_weight")
                    break
        except Exception:
            current_w = None
        if current_w is None:
            current_w = weights.get(sym)

        target_w = _get_targets_weight(targets, sym)
        if target_w is None:
            # fallback from memo.top_drift
            try:
                for row in (memo.top_drift or []):
                    if str(row.get("symbol", "")).upper() == sym:
                        target_w = row.get("target_weight")
                        break
            except Exception:
                target_w = None

        sig_obj = _get_signal_snapshot(signals, sym)
        sig_j = _to_jsonable(sig_obj) if sig_obj is not None else None

        lines = []
        lines.append(f"{sym} is currently in the IBKR portfolio snapshot.")
        if current_w is not None:
            try:
                lines.append(f"- Current weight (approx): {float(current_w)*100:.2f}%")
            except Exception:
                lines.append(f"- Current weight (approx): {current_w}")
        else:
            lines.append("- Current weight: unavailable (market values not available in snapshot).")

        if target_w is None:
            lines.append("- Target weight: unavailable (targets not provided to assistant).")
        else:
            lines.append(f"- Target weight (strategy): {float(target_w)*100:.2f}%")

        if sig_j is None:
            lines.append("- Signals for this symbol: not available in current signals snapshot (common reason a strategy assigns 0%).")
        else:
            # show only existing fields (no fake)
            lines.append("- Signals (available fields):")
            # try common keys
            for k in ["score", "overall_score", "rating", "momentum", "sentiment", "fundamental", "recommendation"]:
                if isinstance(sig_j, dict) and k in sig_j and sig_j.get(k) is not None:
                    lines.append(f"  - {k}: {sig_j.get(k)}")

        # connect to plan/gates context
        if memo.blockers:
            lines.append("")
            lines.append("Note: execution is currently blocked by hard gates, so the system may not be able to act immediately.")
            for b in memo.blockers[:5]:
                lines.append(f"- {b}")

        # If target is 0%, the most honest “reason” is: you held it, but strategy doesn't want it now.
        if target_w is not None and float(target_w) == 0.0:
            lines.append("")
            lines.append("Interpretation (based on platform state):")
            lines.append(f"- You hold {sym} today, but the selected strategy currently assigns it 0% target weight, so it is a candidate to trim/exit when gates allow.")

        return AssistantAnswer(
            ok=True,
            answer="\n".join(lines).strip(),
            evidence={
                "ticker": sym,
                "debug_missing_prices_or_values": sorted(set(missing_mv))[:50],
            } if debug else {},
        )

    # --- general AI: ask anything else -> call research agent with compact context
    compact_ctx = {
        "ts_utc": str(getattr(snapshot, "ts_utc", "") or ""),
        "account": str(getattr(snapshot, "account", "") or ""),
        "strategy_key": strategy_key,
        "blocked": bool(getattr(memo, "blockers", None)),
        "blockers": list(getattr(memo, "blockers", []) or [])[:10],
        "cash_summary": getattr(memo, "cash_summary", None),
        "turnover_summary": getattr(memo, "turnover_summary", None),
        "top_drift": (getattr(memo, "top_drift", None) or [])[:12],
    }

    prompt = (
        "You are the AI Portfolio Manager assistant for a live IBKR-connected platform.\n"
        "Use CONTEXT as source of truth. Do not invent missing values.\n"
        "If the user asks about market/news or whether a holding is good/bad, use your web/news research tools.\n\n"
        "CONTEXT:\n"
        f"{_safe_json(compact_ctx)}\n\n"
        "QUESTION:\n"
        f"{q}\n"
    )

    if enable_research:
        ok, text, ev = _try_research_agent(prompt, history)
        if ok and text:
            return AssistantAnswer(ok=True, answer=text, evidence=ev if debug else {})

        # research failed: show why in debug
        return AssistantAnswer(
            ok=False,
            answer="AI research agent did not return an answer. Enable Debug AI to see the failure reason.",
            evidence=ev if debug else {},
        )

    return AssistantAnswer(ok=True, answer="AI research is disabled.", evidence={})
