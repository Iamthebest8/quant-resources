"""Incremental refresh used by the dashboard button and daily cron.

    python -m pipeline.refresh

.env -> FinMind incremental download (new trading days only) -> local cache ->
recompute latest signals / states / live portfolio with the FROZEN config -> dashboard files.
It never re-selects parameters (no leakage) and never re-downloads full history.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def main() -> None:
    from data.download import download
    from engine.features import compute_features
    from engine.panel import build_panel
    from pipeline.run_research import load_frozen, log, run_pf, run_tl, write_dashboard_data
    from strategy.selection import SignalCache
    from strategy.signals import build_arrays, condition_frames, discovery_score

    if config.IS_SYNTHETIC:
        raise SystemExit("synthetic mode: nothing to refresh")
    log("1/4 FinMind 增量下載（讀取 .env，不顯示 Token）")
    s = download(log=lambda m: log(m))
    log(f"    requests={s.get('requests')} rows={s.get('rows')} bulk_days={s.get('bulk_days', 0)}")
    cfg, frozen = load_frozen("V1")
    if cfg is None:
        raise SystemExit("尚無凍結策略：請先執行 python run_all.py")
    log("2/4 重建 PIT panel 與特徵")
    p = build_panel(log=lambda m: log(m))
    F = compute_features(p, log=lambda m: log(m))
    log(f"3/4 以凍結 {cfg.version} ({frozen['hash']}) 計算最新訊號 / 狀態 / 組合")
    A = build_arrays(p, F)
    sc = SignalCache(p, F)
    conds = condition_frames(p, F, cfg, disc_pct=discovery_score(F, p.universe, cfg.disc_features))
    last = str(p.dates[-1].date())
    _, camps = run_tl(A, sc, cfg, config.RESEARCH_START, last, keep=True)
    pf = run_pf(A, sc, cfg, config.RESEARCH_START, last, positions=True)
    log("4/4 寫出 dashboard 檔案")
    write_dashboard_data(p, F, conds, cfg, A, camps, pf, {})
    log(f"完成：最新資料日期 {last}")


if __name__ == "__main__":
    main()
