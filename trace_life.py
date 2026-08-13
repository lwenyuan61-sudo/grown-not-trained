"""Representative-life trace dump (seed 101, unified protocol): per-trial behavior +
per-block brain state, for the ethogram figure, the HTML replay, and the live brain map.
Output: life_trace.jsonl — one line per trial:
  t, phase, choice, E, hunM, blocks{name: mean activity}, Q4 (H occupancy on race cells),
  teach {da, ht} (delivered console amplitudes).
"""
import json
import numpy as np
from neo_core import Brain, P
import value_confirm2 as U

def traced_life(seed=101):
    P['needs'] = 1; P['hRace'] = 1; P['tauMetab'] = 500.0
    b = Brain(seed)
    names = [k for k in b.blocks.keys()]
    out = open("life_trace.jsonl", "w", encoding="utf-8")
    for t in range(1800):
        poor = 0.08 if t < 1200 else 0.03
        e_before = float(b.energy)
        crossed, hunM, E = U.trial(b, poor, True)
        da = ht = 0.0
        if crossed == U.RICHP: da = min(0.9, U.KBODY * max(0.0, E + 0.002 - e_before))
        if crossed == U.TOXP: da, ht = U.KBODY * U.SWEET, min(0.9, U.KHARM * U.POISON)
        if hunM > 0.5: ht = max(ht, min(0.9, U.KPANG * hunM))
        row = dict(t=t, phase="plenty" if t < 1200 else "winter",
                   choice=int(crossed), E=round(E, 3), hun=round(hunM, 3),
                   blocks={n: round(float(b.act[b.blocks[n]].mean()), 3) for n in names},
                   Q4=round(float(b.Q[4][b.outs].mean()), 3),
                   teach=dict(da=round(da, 2), ht=round(ht, 2)))
        out.write(json.dumps(row) + "\n")
        if t % 200 == 0: print(t, flush=True)
    out.close(); print("done", flush=True)

if __name__ == "__main__":
    traced_life()
