"""SHORT EXECUTABILITY data: probe FinMind margin / short-sale / securities-lending datasets and cache the
per-stock series needed to audit Weinstein short entries.  python -m data.short_data --probe | --ids 2330,2317
The token is read from .env by the client and never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data.finmind_client import FinMindCache, FinMindClient, FinMindError  # noqa: E402

DATASETS = {
    "TaiwanStockMarginPurchaseShortSale": "個股融資融券（含融券餘額、融券限額 ShortSaleLimit、註記 Note）",
    "TaiwanDailyShortSaleBalances": "信用額度總量管制餘額（融券 + 借券賣出餘額）",
    "TaiwanStockSecuritiesLending": "借券成交明細（費率）",
    "TaiwanTotalMarginPurchaseShortSale": "全市場融資融券",
}


def probe(log=print) -> list[dict]:
    client = FinMindClient()
    out = []
    for ds, label in DATASETS.items():
        kw = {"data_id": "2330"} if ds != "TaiwanTotalMarginPurchaseShortSale" else {}
        try:
            df = client.fetch(ds, start_date="2025-03-03", end_date="2025-03-14", validate=False, **kw)
            out.append({"dataset": ds, "label": label, "status": "OK" if len(df) else "EMPTY", "rows": len(df),
                        "columns": list(df.columns)})
        except FinMindError as e:
            out.append({"dataset": ds, "label": label, "status": type(e).__name__, "rows": 0,
                        "error": str(e)[:160]})
        log(f"[short] {ds}: {out[-1]['status']} rows={out[-1]['rows']}")
    (config.OUT_DIR / "phase2").mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "phase2" / "short_data_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                     encoding="utf-8")
    return out


def download(ids: list[str], start: str = "2022-06-01", end: str | None = None, log=print) -> dict:
    cache = FinMindCache(FinMindClient())
    n = 0
    for ds in ("TaiwanStockMarginPurchaseShortSale", "TaiwanDailyShortSaleBalances"):
        for i, sid in enumerate(ids):
            try:
                cache.series(ds, sid, start, end)
                n += 1
            except FinMindError as e:
                log(f"[short] {ds} {sid}: {type(e).__name__}")
            if (i + 1) % 100 == 0:
                cache.save_manifest()
                log(f"[short] {ds} {i + 1}/{len(ids)}")
        cache.save_manifest()
    return {"requests": n}


def load(ds: str, ids: list[str]) -> pd.DataFrame:
    cache = FinMindCache(client=None)
    frames = [cache.read(ds, s) for s in ids]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--ids-file", default=None)
    a = ap.parse_args()
    if a.probe:
        probe()
    if a.ids_file:
        ids = pd.read_csv(a.ids_file, dtype=str)["stock_id"].drop_duplicates().tolist()
        print(download(ids))


if __name__ == "__main__":
    main()
