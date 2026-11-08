"""ONE-CLICK REPRODUCTION — 一键复现
Usage:
    python run_all.py            # quick demo (~3 min): one fully-logged life, watch her live
    python run_all.py full       # full confirmatory campaign (hours): every paper number
Requires: Python >= 3.12, numpy >= 2.0. Nothing else.
"""
import subprocess, sys, time

QUICK = [("trace_life.py", "One fully-logged life (seed 101): plenty then winter. "
          "Output: life_trace.jsonl - every choice, energy, hunger, teaching event.")]
FULL = [
    ("verdict_creb.py", "Part I substrate battery."),
    ("neo_regression.py", "Golden regressions: physics unchanged across additions."),
    ("value_confirm3.py", "Part II main pack: 30 seeds, always-on plasticity, severed twins."),
    ("c1_alwayson.py", "Acquisition from chance, body vs sham wire."),
    ("swap_confirm.py", "Location-swap control: does preference follow the food or the place?"),
    ("discover_confirm.py", "Pressure-rewrite: settled on poor (rich absent), famine hits, rich appears."),
    ("q_baseline.py", "Tabular Q baseline: identical payout, observations and budget."),
    ("discover_trace.py", "Logged discovery life (figure data)."),
]

def run(script, why):
    print(f"\n=== {script} ===\n{why}\n", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, script])
    print(f"--- {script}: exit {r.returncode}, {time.time()-t0:.0f}s ---", flush=True)
    return r.returncode

if __name__ == "__main__":
    jobs = QUICK + (FULL if "full" in sys.argv[1:] else [])
    fails = [s for s, w in jobs if run(s, w) != 0]
    print("\nALL DONE" if not fails else f"\nFAILED: {fails}")
    if "full" not in sys.argv[1:]:
        print("(Quick demo finished. For every paper number: python run_all.py full)")
