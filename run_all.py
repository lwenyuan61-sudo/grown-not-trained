"""ONE-CLICK REPRODUCTION — 一键复现
Usage:
    python run_all.py            # quick demo (~3 min): one fully-logged life, watch her live
    python run_all.py full       # full confirmatory campaign (~3 h): every paper number
Requires: Python >= 3.12, numpy >= 2.0. Nothing else.
"""
import subprocess, sys, time

QUICK = [("trace_life.py", "One fully-logged life (seed 101): plenty then winter. "
          "Output: life_trace.jsonl - every choice, energy, hunger, teaching event.")]
FULL = [
    ("value_confirm2.py", "Unified pack: one machine, one life, six readouts x 10 seeds (+ severed twins)."),
    ("value_confirm.py", "Historical-assay pack incl. the single-food C1 rig with sham-body controls."),
    ("swap_confirm.py", "Location-swap control: does preference follow the food or the place?"),
    ("discover_confirm.py", "Pressure-rewrite: settled on poor (rich absent), famine hits, rich appears."),
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
