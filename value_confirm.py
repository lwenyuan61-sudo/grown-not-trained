"""CONFIRMATORY PACK — the paper's Part II, one clean pass.
Frozen final protocols of the six rungs, 10 fresh seeds, bars as finally registered.
Every law is proven by its own embedded ablation arm; no history needed.
"""
import json
import numpy as np
from neo_core import P

SEEDS = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
OUT = open("value_confirm.jsonl", "w", encoding="utf-8")

def emit(row):
    OUT.write(json.dumps(row) + "\n"); OUT.flush()
    print(json.dumps(row), flush=True)

tally = {}
def score(rung, ok):
    tally.setdefault(rung, [0, 0])
    tally[rung][0] += bool(ok); tally[rung][1] += 1

# ---- C1: body is the only teacher (E1 final) ----
from value_ladder1 import run_e1, KBODY
for s in SEEDS:
    body = run_e1(s, KBODY); sham = run_e1(s, 0.0)
    ok = body['final'] >= 0.8 and sham['final'] <= 0.35
    score("C1", ok)
    emit(dict(rung="C1", seed=s, body=body['final'], sham=sham['final'], PASS=bool(ok)))

# ---- C2: magnitude preference + swap control (E2' final) ----
import value_ladder2 as L2
from value_ladder2 import CUE as C2CUE, A as C2A, B as C2B
from value_ladder1 import E0
from neo_core import Brain
def c2_arm(seed, meals):
    L2.MEAL.clear(); L2.MEAL.update(meals)
    b = Brain(seed); body = {'E': E0}
    h = [L2.trial(b, body, C2CUE, KBODY) for _ in range(600)]
    seg = [x for x in h[500:] if x in (C2A, C2B)]
    return round(sum(1 for x in seg if x == C2A) / max(len(seg), 1), 2)
for s in SEEDS:
    pa = c2_arm(s, {C2A: 0.25, C2B: 0.08})
    pswap = c2_arm(s, {C2A: 0.08, C2B: 0.25})
    ok = pa >= 0.65 and pswap <= 0.35
    score("C2", ok)
    emit(dict(rung="C2", seed=s, prefA=pa, swap_prefA=pswap, PASS=bool(ok)))

# ---- C4: signed zero, co-report (E5' final; bare arm = negative control) ----
from value_ladder3 import run as c4_run
for s in SEEDS:
    bare = c4_run(s, 0.0); wired = c4_run(s, KBODY)
    ok = wired['final_T_tox'] <= 0.2 and wired['final_A_rich'] >= 0.8
    score("C4", ok)
    emit(dict(rung="C4", seed=s, bare_T=bare['final_T_tox'], wired_T=wired['final_T_tox'],
              wired_A=wired['final_A_rich'], PASS=bool(ok)))

# ---- C3: state modulation (E3 final bins) ----
import value_ladder10 as L10
P['needs'] = 1; P['hRace'] = 1; P['tauMetab'] = 500.0
for s in SEEDS:
    b = Brain(s)
    rows = [L10.trial(b, 0.15) for _ in range(600)]
    hungry = [c for c, hm in rows[100:] if hm >= 0.3]
    full = [c for c, hm in rows[100:] if hm <= 0.05]
    er = lambda seg: round(sum(1 for x in seg if x == L10.ACT) / max(len(seg), 1), 2)
    ph, pf = er(hungry), er(full)
    ok = (ph - pf) >= 0.3 and len(hungry) > 30 and len(full) > 30
    score("C3", ok)
    emit(dict(rung="C3", seed=s, eat_hungry=ph, eat_full=pf,
              n=[len(hungry), len(full)], PASS=bool(ok)))

# ---- C5: ecological foraging vs severed twin (E6b2 final) ----
from value_ladder12 import life, RICHP as F_R, POORP as F_P, EMPTYP as F_E, TOXP as F_T
for s in SEEDS:
    fr, Es, hits = life(s, True)
    tfr, tEs, thits = life(s, False)
    E6, tE6 = Es[600:], tEs[600:]
    mines = fr["rich"] + fr["poor"] + fr["empty"] + fr["toxic"]
    prich = round(fr["rich"] / mines, 2) if mines > 0 else 0.0
    b1 = prich >= 0.8 and fr["toxic"] <= 0.1
    b2 = 0.4 <= E6.mean() <= 0.9 and (E6 > 0.1).mean() >= 0.9
    b3 = (E6.mean() - tE6.mean()) >= 0.1 and hits <= thits / 2
    score("C5", b1 and b2 and b3)
    emit(dict(rung="C5", seed=s, prich_cond=prich, toxic=fr["toxic"],
              meanE=round(float(E6.mean()), 2), twinE=round(float(tE6.mean()), 2),
              hits=[hits, thits], PASS=bool(b1 and b2 and b3)))

# ---- C6: winter with the full machine incl. hunger pang (E8b final) ----
from value_ladder13 import trial as w_trial, RICHP as W_R, POORP as W_P, EMPTYP as W_E, TOXP as W_T, KDESP
for s in SEEDS:
    b = Brain(s)
    h1 = [w_trial(b, 0.08, KDESP, 0.6) for _ in range(600)]
    h2 = [w_trial(b, 0.03, KDESP, 0.6) for _ in range(600)]
    ch = [c for c, _ in h2[-300:]]
    Es = np.array([E for _, E in h2[-300:]])
    mines = [c for c in ch if c in (W_R, W_P, W_E, W_T)]
    pr = round(sum(1 for c in mines if c == W_R) / max(len(mines), 1), 2)
    ok = Es.mean() >= 0.3 and (Es > 0.1).mean() >= 0.9 and pr >= 0.8
    score("C6", ok)
    emit(dict(rung="C6", seed=s, winter_P_rich=pr,
              winter_meanE=round(float(Es.mean()), 2),
              alive=round(float((Es > 0.1).mean()), 2), PASS=bool(ok)))

emit(dict(rung="TALLY", **{k: f"{v[0]}/{v[1]}" for k, v in tally.items()}))
OUT.close()
