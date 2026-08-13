"""CONFIRMATORY PACK v3 — the review-repair run. Three corrections, one upgrade:
(1) ALWAYS-ON plasticity: P['alwaysOn']=1 with DAT reuptake clearance (P['datPump']=1).
    No teaching boolean matters any more — every tick of race, ITI and outcome learns
    by the same chemistry. This is the run that either earns or retires the paper's
    "no training phase" claim (Reviewer 2, M1a).
(2) The severed twin is now the STATED control: teaching conductances zeroed
    (kb=kh=kp=0) while plasticity, decay and turnover run identically (R2, M6).
(3) The toxic option's sweetness teaches by ABSORBED delta, like every meal (R2, M5ii).
Upgrade: 30 fresh seeds (first 30 primes > 100), full per-seed JSONL for the SI table.
"""
import json
import numpy as np
from neo_core import Brain, P, SLOTC

RICHP, POORP, EMPTYP, TOXP = 1, 3, 5, 6
KBODY, KHARM, KPANG, KDESP, H0 = 3.2, 3.2, 0.6, 2.0, 0.15
SWEET, POISON, CTX = 0.05, 0.25, 2
SEEDS = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
         179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257]

def trial(b, poor_meal, kb, kh, kp):
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
    hit = False
    if crossed >= 0:
        b.hold[b.outs] = 0.0
        cons = np.zeros(b.N)
        if crossed in (RICHP, POORP):
            e0 = b.energy; b.eat(0.25 if crossed == RICHP else poor_meal)
            cons[b.blocks['pcmp']] = min(0.9, kb * (b.energy - e0))
        elif crossed == TOXP:
            hit = True
            e0 = b.energy
            b.energy = min(1.0, b.energy + SWEET)
            dS = b.energy - e0                       # absorbed, like any meal (R2 M5ii)
            b.energy = max(0.05, b.energy - POISON)
            cons[b.blocks['pcmp']] = min(0.9, kb * dS)
            cons[b.blocks['ncmp']] = min(0.9, kh * POISON)
        if kp > 0 and hunM > 0.5:
            cons[b.blocks['ncmp']] = np.maximum(cons[b.blocks['ncmp']],
                                                min(0.9, kp * hunM))
        # alwaysOn: the boolean below no longer gates plasticity; it only carries the
        # teaching conductances. The twin passes kb=kh=kp=0 and learns by the same
        # physics on zero-valued verdicts.
        if cons.any():
            for _ in range(P['TEACH']): b.tick(cons, True)
        b.adC[crossed] += 1.0
    b.hold[:] = 0.0; b.posOn = False
    for _ in range(P['ITI']): b.tick(z, False)
    return crossed, hunM, float(b.energy), hit

def life(seed, kb, kh, kp):
    b = Brain(seed)
    rows = [trial(b, 0.08, kb, kh, kp) for _ in range(1200)]
    rows += [trial(b, 0.03, kb, kh, kp) for _ in range(600)]
    return rows

def visits(rows, a, z, coh):
    seg = [c for c, _, _, _ in rows[a:z]]
    mines = [c for c in seg if c in (RICHP, POORP, EMPTYP, TOXP)]
    return sum(1 for c in mines if c == coh) / max(len(mines), 1)

if __name__ == "__main__":
    P['needs'] = 1; P['hRace'] = 1; P['tauMetab'] = 500.0
    P['alwaysOn'] = 1; P['datPump'] = 1
    out = open("value_confirm3.jsonl", "w", encoding="utf-8")
    tally = {}
    for seed in SEEDS:
        L = life(seed, KBODY, KHARM, KPANG)
        T = life(seed, 0.0, 0.0, 0.0)
        c1_late = sum(1 for c, _, _, _ in L[900:1200] if c == RICHP) / 300
        t_late = sum(1 for c, _, _, _ in T[900:1200] if c == RICHP) / 300
        pr = visits(L, 900, 1200, RICHP); pp = visits(L, 900, 1200, POORP)
        C2 = (pr / max(pr + pp, 1e-9)) >= 0.8
        hb = [(c, hm) for c, hm, _, _ in L[200:1200]]
        hungry = [c for c, hm in hb if hm >= 0.3]; full = [c for c, hm in hb if hm <= 0.05]
        er = lambda seg: sum(1 for x in seg if x in (RICHP, POORP)) / max(len(seg), 1)
        C3 = (er(hungry) - er(full)) >= 0.3 and len(hungry) > 30 and len(full) > 30
        ltox = sum(1 for r in L if r[3]); ttox = sum(1 for r in T if r[3])
        C4 = visits(L, 900, 1200, TOXP) <= 0.1 and ltox <= ttox / 2
        E5 = np.array([E for _, _, E, _ in L[600:1200]])
        tE5 = np.array([E for _, _, E, _ in T[600:1200]])
        C5 = 0.4 <= E5.mean() <= 0.9 and (E5 > 0.1).mean() >= 0.9 and (E5.mean() - tE5.mean()) >= 0.1
        Ew = np.array([E for _, _, E, _ in L[1500:]])
        prw = visits(L, 1500, 1800, RICHP)
        C6 = Ew.mean() >= 0.3 and (Ew > 0.1).mean() >= 0.9 and prw >= 0.8
        row = dict(seed=seed, c1=[round(c1_late, 2), round(t_late, 2)],
                   c2=round(pr / max(pr + pp, 1e-9), 2),
                   c3=[round(er(hungry), 2), round(er(full), 2), len(hungry), len(full)],
                   c4=[ltox, ttox], c5=[round(float(E5.mean()), 2), round(float(tE5.mean()), 2)],
                   c6=[round(float(Ew.mean()), 2), round(prw, 2)],
                   PASS={k: bool(v) for k, v in
                         zip("C2 C3 C4 C5 C6".split(), (C2, C3, C4, C5, C6))})
        for k, v in row["PASS"].items():
            tally.setdefault(k, [0, 0]); tally[k][0] += v; tally[k][1] += 1
        out.write(json.dumps(row) + "\n"); out.flush()
        print(json.dumps(row), flush=True)
    print(json.dumps({"TALLY": {k: f"{v[0]}/{v[1]}" for k, v in tally.items()},
                      "config": "alwaysOn+datPump, stated twin, absorbed sweet, 30 seeds"}),
          flush=True)
    out.close()
