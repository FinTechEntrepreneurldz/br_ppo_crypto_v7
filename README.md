# br_ppo_crypto_v7

Paper-trading repo for `br_ppo_crypto_v7`.

This repo uses the uploaded V7 BR-PPO ensemble artifacts and writes dashboard-compatible logs for the QSentia investor site.

## GitHub Secrets

Add these under Settings → Secrets and variables → Actions → Secrets:

- `ALPACA_CRYPTO_V7_KEY_ID`
- `ALPACA_CRYPTO_V7_SECRET_KEY`

## GitHub Variables

Add these under Settings → Secrets and variables → Actions → Variables:

- `ALLOCATION_MODE=ppo`
- `DEFAULT_ACTION=v7_symbolic_optuna_00__long_top`
- `SUBMIT_ORDERS=false` first, then `true` after verifying logs
- `CANCEL_OPEN_ORDERS=true` if you want to cancel stale open orders before submitting new ones
- `MIN_ORDER_NOTIONAL=25`
- `REBALANCE_EVERY_DAYS=10`
- `FORCE_REBALANCE=false`

## Verify model usage

Check `logs/decisions/latest_decision.csv` after a run. You want:

- `allocation_mode=ppo`
- `action_source=ppo_ensemble_model`
- `fallback_used=false`

If fallback is true, the reason is logged in `fallback_reason`.
