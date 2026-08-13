"""Pressure-rewrite confirm (C6b): in plenty the rich mine DOES NOT EXIST — settling on
poor is the legitimate optimum, not a pathology. At winter the poor mine thins below
metabolic cost and the rich mine appears. The settled habit must be rewritten under
pressure: desperation (shake) + hunger-pang (chisel) + a new option to discover.
10 fresh seeds. BARS (last 300 of winter): P(rich|visit) >= 0.8 and mean E >= 0.3.
"""
import json
import numpy as np
from neo_core import Brain, P, SLOTC
import value_confirm2 as U

def trial(b, rich_meal, poor_meal):
    z = np.zeros(b.N)
    b.trial_reset(); b.encode_seq([U.CTX])
    for _ in range(4): b.tick(z, False)
    b.pos = SLOTC[0]
    b.goI[b.outs] = np.minimum(b.goI[b.outs], 0)
    b.hold[:] = 0; b.tick(z, False)
    hunM = float(b.act[b.blocks["hun"]].mean())
    sig = P["sigmaCoh"] * (1.0 + U.KDESP * hunM)
    dec = np.exp(-1 / P["tauOU"]); kn = np.sqrt(1 - dec * dec)
    nz = sig * b.rng.standard_normal(b.nSym)
    crossed = -1
    for t in range(P["MAXW"]):
        nz = nz * dec + sig * kn * b.rng.standard_normal(b.nSym)
        gain = U.H0 + (1.0 - U.H0) * np.clip(P["hGain"] * b.Q[4][b.outs], 0, 1)
        b.hold[b.outs] = gain * (nz[b.cohOf[b.outs]]
                                 + P["sigmaOut"] * b.rng.standard_normal(len(b.outs)))
        b.tick(z, False)
        means = [b.cohort_mean(c) for c in range(b.nSym)]
        mx = int(np.argmax(means))
        if means[mx] > P["THETA_EMIT"]:
            crossed = mx; break
    if crossed >= 0:
        b.hold[b.outs] = 0.0
        cons = np.zeros(b.N)
        meal = rich_meal if crossed == U.RICHP else (poor_meal if crossed == U.POORP else 0.0)
        if meal > 0:
            e0 = b.energy; b.eat(meal)
            cons[b.blocks["pcmp"]] = min(0.9, U.KBODY * (b.energy - e0))
        elif crossed == U.TOXP:
            b.energy = min(1.0, b.energy + U.SWEET)
            b.energy = max(0.05, b.energy - U.POISON)
            cons[b.blocks["pcmp"]] = min(0.9, U.KBODY * U.SWEET)
            cons[b.blocks["ncmp"]] = min(0.9, U.KHARM * U.POISON)
        if U.KPANG > 0 and hunM > 0.5:
            cons[b.blocks["ncmp"]] = np.maximum(cons[b.blocks["ncmp"]],
                                                min(0.9, U.KPANG * hunM))
        if cons.any():
            for _ in range(P["TEACH"]): b.tick(cons, True)
        b.adC[crossed] += 1.0
    b.hold[:] = 0.0; b.posOn = False
    for _ in range(P["ITI"]): b.tick(z, False)
    return crossed, float(b.energy)

if __name__ == "__main__":
    P["needs"] = 1; P["hRace"] = 1; P["tauMetab"] = 500.0
    P["alwaysOn"] = 1; P["datPump"] = 1
    out = open("discover_confirm.jsonl", "w", encoding="utf-8")
    passes = 0
    for seed in U.SEEDS:
        b = Brain(seed)
        h1 = [trial(b, 0.0, 0.08) for _ in range(1200)]   # plenty: NO rich mine
        h2 = [trial(b, 0.25, 0.03) for _ in range(600)]   # winter: poor starves, rich appears
        def stats(rows):
            ch = [c for c, _ in rows[-300:]]
            Es = np.array([E for _, E in rows[-300:]])
            mines = [c for c in ch if c in (U.RICHP, U.POORP, U.EMPTYP, U.TOXP)]
            pr = sum(1 for c in mines if c == U.RICHP) / max(len(mines), 1)
            pp = sum(1 for c in mines if c == U.POORP) / max(len(mines), 1)
            return round(pr, 2), round(pp, 2), round(float(Es.mean()), 2)
        pr1, pp1, e1 = stats(h1)
        pr2, pp2, e2 = stats(h2)
        ok = pr2 >= 0.8 and e2 >= 0.3
        passes += ok
        row = dict(seed=seed, plenty=dict(P_poor=pp1, meanE=e1),
                   winter=dict(P_rich=pr2, P_poor=pp2, meanE=e2), PASS=bool(ok))
        out.write(json.dumps(row) + "\n"); out.flush()
        print(json.dumps(row), flush=True)
    print(json.dumps(dict(TALLY=f"{passes}/10")), flush=True)
    out.close()
