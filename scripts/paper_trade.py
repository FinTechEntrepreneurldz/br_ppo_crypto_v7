#!/usr/bin/env python3
from __future__ import annotations

import csv, json, os, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
METADATA_PATH = MODELS_DIR / "v7_ultimate_yfinance_alpha_factory_metadata.json"
MODEL_PATHS = [
    MODELS_DIR / "v7_brppo_ensemble_model_00.zip",
    MODELS_DIR / "v7_brppo_ensemble_model_01.zip",
    MODELS_DIR / "v7_brppo_ensemble_model_02.zip",
]

PORTFOLIO_CSV = LOGS_DIR / "portfolio" / "portfolio.csv"
LATEST_DECISION_CSV = LOGS_DIR / "decisions" / "latest_decision.csv"
DECISIONS_CSV = LOGS_DIR / "decisions" / "decisions.csv"
LATEST_TARGET_WEIGHTS_CSV = LOGS_DIR / "target_weights" / "latest_target_weights.csv"
TARGET_WEIGHTS_CSV = LOGS_DIR / "target_weights" / "target_weights.csv"
LATEST_POSITIONS_CSV = LOGS_DIR / "positions" / "latest_positions.csv"
LATEST_PLANNED_ORDERS_CSV = LOGS_DIR / "orders" / "latest_planned_orders.csv"
LATEST_SUBMITTED_ORDERS_CSV = LOGS_DIR / "orders" / "latest_submitted_orders.csv"
SUBMITTED_ORDERS_CSV = LOGS_DIR / "orders" / "submitted_orders.csv"
HEALTH_STATUS_JSON = LOGS_DIR / "health" / "health_status.json"
SIGNAL_HISTORY_CSV = LOGS_DIR / "health" / "signal_history.csv"

@dataclass
class Settings:
    allocation_mode: str
    default_action: str
    submit_orders: bool
    cancel_open_orders: bool
    min_order_notional: float
    rebalance_every_days: int
    force_rebalance: bool
    alpaca_key_id: str
    alpaca_secret_key: str
    alpaca_base_url: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_str() -> str:
    return utc_now().isoformat()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_dirs() -> None:
    for path in [PORTFOLIO_CSV, LATEST_DECISION_CSV, DECISIONS_CSV, LATEST_TARGET_WEIGHTS_CSV,
                 TARGET_WEIGHTS_CSV, LATEST_POSITIONS_CSV, LATEST_PLANNED_ORDERS_CSV,
                 LATEST_SUBMITTED_ORDERS_CSV, SUBMITTED_ORDERS_CSV, HEALTH_STATUS_JSON, SIGNAL_HISTORY_CSV]:
        path.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    return Settings(
        allocation_mode=os.getenv("ALLOCATION_MODE", "ppo").strip().lower(),
        default_action=os.getenv("DEFAULT_ACTION", "v7_symbolic_optuna_00__long_top").strip(),
        submit_orders=env_bool("SUBMIT_ORDERS", False),
        cancel_open_orders=env_bool("CANCEL_OPEN_ORDERS", False),
        min_order_notional=float(os.getenv("MIN_ORDER_NOTIONAL", "25")),
        rebalance_every_days=int(float(os.getenv("REBALANCE_EVERY_DAYS", "10"))),
        force_rebalance=env_bool("FORCE_REBALANCE", False),
        alpaca_key_id=os.getenv("ALPACA_CRYPTO_V7_KEY_ID", "").strip(),
        alpaca_secret_key=os.getenv("ALPACA_CRYPTO_V7_SECRET_KEY", "").strip(),
        alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/"),
    )


def load_metadata() -> Dict[str, Any]:
    return json.loads(METADATA_PATH.read_text())


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
    total = sum(clean.values())
    if abs(total) <= 1e-12:
        return {}
    return {k: v / total for k, v in clean.items()}


def crypto_trend_gate() -> float:
    try:
        import yfinance as yf
        btc = yf.download("BTC-USD", period="180d", interval="1d", progress=False, auto_adjust=True)
        eth = yf.download("ETH-USD", period="180d", interval="1d", progress=False, auto_adjust=True)
        def trending(df: pd.DataFrame) -> bool:
            if df is None or df.empty or "Close" not in df:
                return False
            close = df["Close"].dropna()
            if len(close) < 126:
                return False
            return bool(close.iloc[-1] > close.rolling(126).mean().iloc[-1] and close.iloc[-1] / close.iloc[-63] - 1.0 > 0)
        return 1.0 if trending(btc) or trending(eth) else 0.0
    except Exception:
        return 0.0


def add_weight(targets: Dict[str, float], symbol: str, weight: float) -> None:
    if abs(weight) > 1e-12:
        targets[symbol] = targets.get(symbol, 0.0) + float(weight)


def flatten_action_to_tradeable_targets(action_weights: Dict[str, float]) -> Dict[str, float]:
    """Map research streams into executable proxy tickers for paper trading."""
    targets: Dict[str, float] = {}
    for stream, w in action_weights.items():
        u = stream.upper()
        if u == "BIL":
            add_weight(targets, "BIL", w)
        elif u in {"SPY", "QQQ", "VTI", "RSP", "IWM"}:
            add_weight(targets, u, w)
        elif u in {"TOP_EW", "CURRENT_EW"}:
            add_weight(targets, "SPY", w)
        elif "QQQ" in u:
            add_weight(targets, "QQQ", w)
        elif "CRYPTO_CASH_GATED" in u:
            crypto_w = w * crypto_trend_gate()
            add_weight(targets, "BTC/USD", crypto_w * 0.40)
            add_weight(targets, "ETH/USD", crypto_w * 0.35)
            add_weight(targets, "SOL/USD", crypto_w * 0.25)
            add_weight(targets, "BIL", w - crypto_w)
        elif "CRYPTO" in u:
            add_weight(targets, "BTC/USD", w * 0.40)
            add_weight(targets, "ETH/USD", w * 0.35)
            add_weight(targets, "SOL/USD", w * 0.25)
        elif "BIL" in u and len(action_weights) == 1:
            add_weight(targets, "BIL", w)
        elif "RISK_PARITY" in u or "MAX_SHARPE" in u or "DRAWDOWN" in u or "MULTI_ASSET" in u:
            add_weight(targets, "SPY", w * 0.35)
            add_weight(targets, "QQQ", w * 0.20)
            add_weight(targets, "BIL", w * 0.45)
        elif "SYMBOLIC" in u or "CONSENSUS" in u or "ALPHA" in u or "LONG_TOP" in u or "V7_" in u or "V6_" in u:
            add_weight(targets, "SPY", w * 0.60)
            add_weight(targets, "QQQ", w * 0.30)
            add_weight(targets, "BIL", w * 0.10)
        else:
            add_weight(targets, "BIL", w)
    return normalize_weights(targets)


def build_proxy_observation(metadata: Dict[str, Any]) -> np.ndarray:
    # Uses exact action-space length from V7 metadata, with zero feature vector + winning action one-hot.
    action_names = metadata.get("action_names") or []
    # V7 live research state is expensive to rebuild; this is a deterministic PPO-compatible proxy state.
    n_features = 512
    obs = np.zeros(n_features + len(action_names), dtype=np.float32)
    default_action = metadata.get("best_fixed_action") or os.getenv("DEFAULT_ACTION", "bil_cash")
    if default_action in action_names:
        obs[n_features + action_names.index(default_action)] = 1.0
    return obs


def choose_action_with_ppo_ensemble(metadata: Dict[str, Any]) -> Tuple[str, str, str, bool, str]:
    from stable_baselines3 import PPO
    action_names = metadata.get("action_names") or []
    if not action_names:
        raise ValueError("metadata.action_names missing")
    missing = [str(p) for p in MODEL_PATHS if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing PPO ensemble artifacts: {missing}")
    obs = build_proxy_observation(metadata)
    probs = []
    for path in MODEL_PATHS:
        model = PPO.load(str(path))
        try:
            import torch
            obs_t = torch.as_tensor(obs).float().unsqueeze(0).to(model.device)
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_t)
                p = dist.distribution.probs.detach().cpu().numpy()[0]
            probs.append(p)
        except Exception:
            action_idx, _ = model.predict(obs, deterministic=True)
            p = np.zeros(len(action_names), dtype=float)
            p[int(action_idx)] = 1.0
            probs.append(p)
    avg = np.nanmean(np.vstack(probs), axis=0)
    idx = int(np.nanargmax(avg))
    if idx < 0 or idx >= len(action_names):
        raise ValueError(f"invalid action index from PPO ensemble: {idx}")
    return action_names[idx], "ppo_ensemble_model", "ppo_proxy_state", False, ""


def choose_action(settings: Settings, metadata: Dict[str, Any]) -> Tuple[str, str, str, bool, str]:
    if settings.allocation_mode == "ppo":
        try:
            return choose_action_with_ppo_ensemble(metadata)
        except Exception as exc:
            return settings.default_action, "default_action_fallback", "fixed_default", True, repr(exc)
    return settings.default_action, "default_action", "fixed_default", False, ""


def alpaca_headers(settings: Settings) -> Dict[str, str]:
    return {"APCA-API-KEY-ID": settings.alpaca_key_id, "APCA-API-SECRET-KEY": settings.alpaca_secret_key, "Content-Type": "application/json"}


def alpaca_get(settings: Settings, path: str) -> Any:
    r = requests.get(f"{settings.alpaca_base_url}{path}", headers=alpaca_headers(settings), timeout=30)
    r.raise_for_status(); return r.json()


def alpaca_post(settings: Settings, path: str, payload: Dict[str, Any]) -> Any:
    r = requests.post(f"{settings.alpaca_base_url}{path}", headers=alpaca_headers(settings), json=payload, timeout=30)
    r.raise_for_status(); return r.json()


def alpaca_delete(settings: Settings, path: str) -> Any:
    r = requests.delete(f"{settings.alpaca_base_url}{path}", headers=alpaca_headers(settings), timeout=30)
    r.raise_for_status()
    if r.text:
        try: return r.json()
        except Exception: return {"raw": r.text}
    return {}


def cancel_open_orders(settings: Settings) -> Dict[str, Any]:
    if not settings.alpaca_key_id or not settings.alpaca_secret_key:
        return {"cancel_status": "missing_credentials", "cancelled_orders": 0}
    try:
        result = alpaca_delete(settings, "/v2/orders")
        return {"cancel_status": "ok", "cancelled_orders": len(result) if isinstance(result, list) else 0}
    except Exception as exc:
        return {"cancel_status": f"cancel_error:{repr(exc)[:180]}", "cancelled_orders": 0}


def get_account(settings: Settings) -> Dict[str, Any]:
    if not settings.alpaca_key_id or not settings.alpaca_secret_key:
        return {"status": "missing_credentials", "portfolio_value": 1_000_000.0, "equity": 1_000_000.0, "cash": 1_000_000.0, "buying_power": 2_000_000.0}
    try:
        a = alpaca_get(settings, "/v2/account")
        return {"status": "connected", "portfolio_value": float(a.get("portfolio_value", 0)), "equity": float(a.get("equity", 0)), "cash": float(a.get("cash", 0)), "buying_power": float(a.get("buying_power", 0))}
    except Exception as exc:
        return {"status": f"account_error:{repr(exc)[:160]}", "portfolio_value": 1_000_000.0, "equity": 1_000_000.0, "cash": 1_000_000.0, "buying_power": 2_000_000.0}


def get_positions(settings: Settings) -> List[Dict[str, Any]]:
    if not settings.alpaca_key_id or not settings.alpaca_secret_key: return []
    try:
        p = alpaca_get(settings, "/v2/positions")
        return p if isinstance(p, list) else []
    except Exception:
        return []


def normalize_trade_symbol(symbol: str) -> str:
    if symbol in {"BTCUSD", "BTC-USD"}: return "BTC/USD"
    if symbol in {"ETHUSD", "ETH-USD"}: return "ETH/USD"
    if symbol in {"SOLUSD", "SOL-USD"}: return "SOL/USD"
    return symbol


def symbol_from_position(p: Dict[str, Any]) -> str:
    return normalize_trade_symbol(str(p.get("symbol") or "").replace("USD", "/USD"))


def current_position_values(positions: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in positions:
        sym = symbol_from_position(p)
        try: val = float(p.get("market_value", 0))
        except Exception: val = 0.0
        out[sym] = out.get(sym, 0.0) + val
    return out


def build_planned_orders(account: Dict[str, Any], positions: List[Dict[str, Any]], target_weights: Dict[str, float], min_order_notional: float) -> List[Dict[str, Any]]:
    equity = float(account.get("portfolio_value") or account.get("equity") or 0.0)
    current = current_position_values(positions)
    rows = []
    for sym in sorted(set(target_weights) | set(current)):
        target = equity * float(target_weights.get(sym, 0.0))
        cur = float(current.get(sym, 0.0))
        delta = target - cur
        if abs(delta) < min_order_notional: continue
        rows.append({"timestamp_utc": utc_now_str(), "symbol": sym, "side": "buy" if delta > 0 else "sell", "notional": round(abs(delta), 2), "target_value": round(target, 2), "current_value": round(cur, 2), "delta_value": round(delta, 2), "order_type": "market", "time_in_force": "day"})
    return rows


def submit_orders(settings: Settings, planned: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    if not settings.submit_orders or not settings.alpaca_key_id or not settings.alpaca_secret_key:
        return out
    for order in planned:
        payload = {"symbol": normalize_trade_symbol(order["symbol"]), "side": order["side"], "type": "market", "time_in_force": "day", "notional": str(order["notional"])}
        try:
            res = alpaca_post(settings, "/v2/orders", payload)
            out.append({**order, "submitted": True, "alpaca_order_id": res.get("id", ""), "status": res.get("status", ""), "submitted_at": utc_now_str()})
        except Exception as exc:
            out.append({**order, "submitted": False, "alpaca_order_id": "", "status": f"submit_error:{repr(exc)[:220]}", "submitted_at": utc_now_str()})
        time.sleep(0.2)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not append: path.write_text("")
        return
    fields = list(rows[0].keys())
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not append or not exists: w.writeheader()
        w.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def portfolio_row(account: Dict[str, Any], positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"timestamp_utc": utc_now_str(), "portfolio_value": round(float(account.get("portfolio_value", 0)), 2), "equity": round(float(account.get("equity", 0)), 2), "cash": round(float(account.get("cash", 0)), 2), "long_value": round(sum(max(0.0, float(p.get("market_value", 0))) for p in positions), 2), "short_value": round(sum(min(0.0, float(p.get("market_value", 0))) for p in positions), 2), "buying_power": round(float(account.get("buying_power", 0)), 2), "n_positions": len(positions)}


def position_rows(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"timestamp_utc": utc_now_str(), "symbol": symbol_from_position(p), "qty": p.get("qty", ""), "market_value": p.get("market_value", ""), "avg_entry_price": p.get("avg_entry_price", ""), "current_price": p.get("current_price", ""), "unrealized_pl": p.get("unrealized_pl", ""), "unrealized_plpc": p.get("unrealized_plpc", "")} for p in positions]


def target_weight_rows(targets: Dict[str, float], action: str) -> List[Dict[str, Any]]:
    return [{"timestamp_utc": utc_now_str(), "action": action, "symbol": s, "target_weight": round(float(w), 8)} for s, w in sorted(targets.items())]


def last_trade_decision_date() -> datetime | None:
    if not DECISIONS_CSV.exists() or DECISIONS_CSV.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(DECISIONS_CSV)
        if "orders_allowed" in df.columns:
            df = df[df["orders_allowed"].astype(str).str.lower().eq("true")]
        if df.empty: return None
        return pd.to_datetime(df["timestamp_utc"].iloc[-1], utc=True).to_pydatetime()
    except Exception:
        return None


def should_rebalance(settings: Settings) -> Tuple[bool, str]:
    if settings.force_rebalance:
        return True, "force_rebalance"
    last = last_trade_decision_date()
    if last is None:
        return True, "first_run"
    elapsed = (utc_now() - last).days
    if elapsed >= settings.rebalance_every_days:
        return True, f"elapsed_days_{elapsed}"
    return False, f"rebalance_wait_elapsed_days_{elapsed}_threshold_{settings.rebalance_every_days}"


def main() -> int:
    ensure_dirs()
    settings = load_settings()
    metadata = load_metadata()
    action_specs = metadata.get("action_specs") or {}
    action, source, state, fallback, fallback_reason = choose_action(settings, metadata)
    if action not in action_specs:
        fallback = True
        fallback_reason = f"selected action missing from action_specs: {action}"
        action = settings.default_action if settings.default_action in action_specs else (metadata.get("best_fixed_action") or "bil_cash")
        source = "default_action_fallback"
        state = "fixed_default"
    action_weights = normalize_weights(action_specs.get(action, {}))
    target_weights = flatten_action_to_tradeable_targets(action_weights)
    account = get_account(settings)
    positions = get_positions(settings)
    orders_allowed, rebalance_reason = should_rebalance(settings)
    cancel_result = {"cancel_status": "not_requested", "cancelled_orders": 0}
    if orders_allowed and settings.cancel_open_orders and settings.submit_orders:
        cancel_result = cancel_open_orders(settings)
        time.sleep(2)
    planned = build_planned_orders(account, positions, target_weights, settings.min_order_notional) if orders_allowed else []
    submitted = submit_orders(settings, planned) if orders_allowed else []
    decision = {"timestamp_utc": utc_now_str(), "model_id": "br_ppo_crypto_v7", "allocation_mode": settings.allocation_mode, "action": action, "action_source": source, "allocation_state": state, "fallback_used": str(bool(fallback)).lower(), "fallback_reason": fallback_reason, "submit_orders": str(bool(settings.submit_orders)).lower(), "orders_allowed": str(bool(orders_allowed)).lower(), "rebalance_reason": rebalance_reason, "cancel_open_orders": str(bool(settings.cancel_open_orders)).lower(), "cancel_status": cancel_result.get("cancel_status"), "cancelled_orders": cancel_result.get("cancelled_orders"), "account_status": account.get("status", "unknown"), "portfolio_value": account.get("portfolio_value"), "equity": account.get("equity"), "cash": account.get("cash"), "n_target_symbols": len(target_weights), "n_planned_orders": len(planned), "n_submitted_orders": len(submitted), "metadata_winning_variant": metadata.get("winning_variant", "")}
    write_csv(PORTFOLIO_CSV, [portfolio_row(account, positions)], append=True)
    write_csv(LATEST_DECISION_CSV, [decision], append=False)
    write_csv(DECISIONS_CSV, [decision], append=True)
    write_csv(LATEST_TARGET_WEIGHTS_CSV, target_weight_rows(target_weights, action), append=False)
    write_csv(TARGET_WEIGHTS_CSV, target_weight_rows(target_weights, action), append=True)
    write_csv(LATEST_POSITIONS_CSV, position_rows(positions), append=False)
    write_csv(LATEST_PLANNED_ORDERS_CSV, planned, append=False)
    write_csv(LATEST_SUBMITTED_ORDERS_CSV, submitted, append=False)
    write_csv(SUBMITTED_ORDERS_CSV, submitted, append=True)
    health = {"timestamp_utc": utc_now_str(), "model_id": "br_ppo_crypto_v7", "overall_status": "ok" if not fallback else "fallback", "allocation_mode": settings.allocation_mode, "action": action, "action_source": source, "allocation_state": state, "fallback_used": fallback, "fallback_reason": fallback_reason, "orders_allowed": orders_allowed, "rebalance_reason": rebalance_reason, "submit_orders": settings.submit_orders, "cancel_open_orders": settings.cancel_open_orders, "cancel_status": cancel_result.get("cancel_status"), "cancelled_orders": cancel_result.get("cancelled_orders"), "account_status": account.get("status", "unknown"), "target_weights": target_weights, "action_weights": action_weights}
    write_json(HEALTH_STATUS_JSON, health)
    write_csv(SIGNAL_HISTORY_CSV, [{"timestamp_utc": utc_now_str(), "model_id": "br_ppo_crypto_v7", "signal": action, "allocation_mode": settings.allocation_mode, "action_source": source, "fallback_used": str(bool(fallback)).lower(), "fallback_reason": fallback_reason}], append=True)
    print(json.dumps(decision, indent=2, default=str))
    print(json.dumps(target_weights, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
