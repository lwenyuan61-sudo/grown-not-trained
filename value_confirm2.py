"""UNIFIED CONFIRMATORY PACK — one machine, one life, six readouts.
Full final machine (3 wires + brake + desperation), 1200 plenty + 600 winter, vs severed
twin (identical brain/body/ticks, teaching lines cut). 10 fresh seeds. All six paper
claims are metrics of this single protocol.
"""
import json
import numpy as np
from neo_core import Brain, P, SLOTC

RICHP, POORP, EMPTYP, TOXP = 1, 3, 5, 6
KBODY, KHARM, KPANG, KDESP, H0 = 3.2, 3.2, 0.6, 2.0, 0.15
SWEET, POISON, CTX = 0.05, 0.25, 2
SEEDS = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]

def trial(b, poor_meal, learn):
    z = np.zeros(b.N)
    b.trial_reset(); b.encode_seq([CTX])
    for _ in range(4): b.tick(z, False)
    b.pos = SLOTC[0]
    b.goI[b.outs] = np.minimum(b.goI[b.outs], 0)
    b.hold[:] = 0; b.tick(z, False)
    hunM = float(b.act[b.blocks['hun']].mean())
    sig = P['sigmaCoh'] * (1.0 + KDESP * hunM)
    dec = np.exp(-1 / P['tauOU']); kn = np.sqrt(1 - dec * dec)
    nz = sig * b.rng.standard_normal(b.nSym)
    crossed = -1
    for t in range(P['MAXW']):
        nz = nz * dec + sig * kn * b.rng.standard_normal(b.nSym)
        gain = H0 + (1.0 - H0) * np.clip(P['hGain'] * b.Q[4][b.outs], 0, 1)
        b.hold[b.outs] = gain * (nz[b.cohOf[b.outs]]
                                 + P['sigmaOut'] * b.rng.standard_normal(len(b.outs)))
        b.tick(z, False)
        means = [b.cohort_mean(c) for c in range(b.nSym)]
        mx = int(np.argmax(means))
        if means[mx] > P['THETA_EMIT']:
            crossed = mx; break
    if crossed >= 0:
        b.hold[b.outs] = 0.0
        cons = np.zeros(b.N)
        if crossed in (RICHP, POORP):
            e0 = b.energy; b.eat(0.25 if crossed == RICHP else poor_meal)
            cons[b.blocks['pcmp']] = min(0.9, KBODY * (b.energy - e0))
        elif crossed == TOXP:
            b.energy = min(1.0, b.energy + SWEET)
            b.energy = max(0.05, b.energy - POISON)
            cons[b.blocks['pcmp']] = min(0.9, KBODY * SWEET)
            cons[b.blocks['ncmp']] = min(0.9, KHARM * POISON)
        if KPANG > 0 and hunM > 0.5:
            cons[b.blocks['ncmp']] = np.maximum(cons[b.blocks['ncmp']],
                                                min(0.9, KPANG * hunM))
        if learn and cons.any():
            for _ in range(P['TEACH']): b.tick(cons, True)
        elif cons.any():
            for _ in range(P['TEACH']): b.tick(np.zeros(b.N), False)
        b.adC[crossed] += 1.0
    b.hold[:] = 0.0; b.posOn = False
    for _ in range(P['ITI']): b.tick(z, False)
    return crossed, hunM, float(b.energy)

def life(seed, learn):
    b = Brain(seed)
    rows = [trial(b, 0.08, learn) for _ in range(1200)]
    rows += [trial(b, 0.03, learn) for _ in range(600)]
    return rows

def visits(rows, a, z, coh):
    seg = [c for c, _, _ in rows[a:z]]
    mines = [c for c in seg if c in (RICHP, POORP, EMPTYP, TOXP)]
    n = sum(1 for c in mines if c == coh)
    return n / max(len(mines), 1), len(mines), len(seg)

if __name__ == "__main__":
    P['needs'] = 1; P['hRace'] = 1; P['tauMetab'] = 500.0
    out = open("value_confirm2.jsonl", "w", encoding="utf-8")
    tally = {}
    for seed in SEEDS:
        L = life(seed, True)
        T = life(seed, False)
        # C1 learn-from-chance: rich visit share (all trials) first 100 vs plenty-final 300
        c1_early = sum(1 for c, _, _ in L[:100] if c == RICHP) / 100
        c1_late = sum(1 for c, _, _ in L[900:1200] if c == RICHP) / 300
        c1_twin = sum(1 for c, _, _ in T[900:1200] if c == RICHP) / 300
        C1 = c1_late >= 0.3 and c1_late >= 3 * max(c1_early, 0.02) and c1_twin <= 0.2
        # C2 preference among paying mines (plenty final 300)
        pr, _, _ = visits(L, 900, 1200, RICHP); pp, _, _ = visits(L, 900, 1200, POORP)
        C2 = (pr / max(pr + pp, 1e-9)) >= 0.8
        # C3 state modulation within plenty (trials 200-1200)
        hb = [(c, hm) for c, hm, _ in L[200:1200]]
        hungry = [c for c, hm in hb if hm >= 0.3]; full = [c for c, hm in hb if hm <= 0.05]
        er = lambda seg: sum(1 for x in seg if x in (RICHP, POORP)) / max(len(seg), 1)
        C3 = (er(hungry) - er(full)) >= 0.3 and len(hungry) > 30 and len(full) > 30
        # C4 poison refusal (whole life) vs twin
        ltox = sum(1 for c, _, _ in L if c == TOXP); ttox = sum(1 for c, _, _ in T if c == TOXP)
        ptox, _, _ = visits(L, 900, 1200, TOXP)
        C4 = ptox <= 0.1 and ltox <= ttox / 2
        # C5 physiology vs twin (plenty last 600)
        E5 = np.array([E for _, _, E in L[600:1200]]); tE5 = np.array([E for _, _, E in T[600:1200]])
        C5 = 0.4 <= E5.mean() <= 0.9 and (E5 > 0.1).mean() >= 0.9 and (E5.mean() - tE5.mean()) >= 0.1
        # C6 winter (last 300)
        Ew = np.array([E for _, _, E in L[1500:]])
        prw, _, _ = visits(L, 1500, 1800, RICHP)
        C6 = Ew.mean() >= 0.3 and (Ew > 0.1).mean() >= 0.9 and prw >= 0.8
        row = dict(seed=seed, C1=[round(c1_early, 2), round(c1_late, 2), round(c1_twin, 2)],
                   C2=round(pr / max(pr + pp, 1e-9), 2),
                   C3=[round(er(hungry), 2), round(er(full), 2)],
                   C4=[round(ptox, 2), ltox, ttox],
                   C5=[round(float(E5.mean()), 2), round(float(tE5.mean()), 2)],
                   C6=[round(float(Ew.mean()), 2), round(prw, 2)],
                   PASS={k: bool(v) for k, v in
                         zip("C1 C2 C3 C4 C5 C6".split(), (C1, C2, C3, C4, C5, C6))})
        for k, v in row["PASS"].items():
            tally.setdefault(k, [0, 0]); tally[k][0] += v; tally[k][1] += 1
        out.write(json.dumps(row) + "\n"); out.flush()
        print(json.dumps(row), flush=True)
    print(json.dumps({"TALLY": {k: f"{v[0]}/{v[1]}" for k, v in tally.items()}}), flush=True)
    out.close()
