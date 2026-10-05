"""One command: FinMind incremental download -> PIT dataset -> research -> strategy -> backtests -> reports -> dashboard data.

    python run_all.py                 # real FinMind data (needs .env with FINMIND_TOKEN)
    python run_all.py --skip-download # use the local cache only
    EL_DATA_SOURCE=synthetic python run_all.py   # offline code test with synthetic data
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--skip-download", action="store_true")
a = ap.parse_args()
synthetic = os.getenv("EL_DATA_SOURCE", "finmind").lower() == "synthetic"
steps = []
if synthetic:
    steps.append([sys.executable, "-m", "data.synthetic"])
elif not a.skip_download:
    steps.append([sys.executable, "-m", "data.download", "--case-intraday"])
steps.append([sys.executable, "-m", "pipeline.run_research"])
for cmd in steps:
    print(">>", " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
print("完成。啟動 Dashboard：streamlit run dashboard/app.py")
