"""Full-machine SWAP confirm: meals exchanged between locations (the 0.25 meal now at
the 'poor' place). If preference follows the FOOD, lives lock the poor-place cohort.
10 fresh seeds, plenty phase (1200 trials). Bar: P(foodplace|paying visits) >= 0.8, >=8/10.
"""
import json
import numpy as np
from neo_core import Brain, P, SLOTC
import value_confirm2 as U

def swapped_trial(b, learn):
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
        if crossed in (U.RICHP, U.POORP):
            e0 = b.energy
            b.eat(0.08 if crossed == U.RICHP else 0.25)  # SWAPPED meals
            cons[b.blocks["pcmp"]] = min(0.9, U.KBODY * (b.energy - e0))
        elif crossed == U.TOXP:
            b.energy = min(1.0, b.energy + U.SWEET)
            b.energy = max(0.05, b.energy - U.POISON)
            cons[b.blocks["pcmp"]] = min(0.9, U.KBODY * U.SWEET)
            cons[b.blocks["ncmp"]] = min(0.9, U.KHARM * U.POISON)
        if U.KPANG > 0 and hunM > 0.5:
            cons[b.blocks["ncmp"]] = np.maximum(cons[b.blocks["ncmp"]],
                                                min(0.9, U.KPANG * hunM))
        if learn and cons.any():
            for _ in range(P["TEACH"]): b.tick(cons, True)
        b.adC[crossed] += 1.0
    b.hold[:] = 0.0; b.posOn = False
    for _ in range(P["ITI"]): b.tick(z, False)
    return crossed

if __name__ == "__main__":
    P["needs"] = 1; P["hRace"] = 1; P["tauMetab"] = 500.0
    P["alwaysOn"] = 1; P["datPump"] = 1
    passes = 0
    for seed in U.SEEDS:
        b = Brain(seed)
        h = [swapped_trial(b, True) for _ in range(1200)]
        seg = [c for c in h[900:] if c in (U.RICHP, U.POORP)]
        pp = sum(1 for c in seg if c == U.POORP) / max(len(seg), 1)
        ok = pp >= 0.8
        passes += ok
        print(json.dumps(dict(seed=seed, P_food_place=round(pp, 2), PASS=bool(ok))), flush=True)
    print(json.dumps(dict(TALLY=f"{passes}/10")), flush=True)
