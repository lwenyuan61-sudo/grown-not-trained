"""FINAL VERDICT: CREB-GATE config (v3: slow nuclear integrator gates capture, anchor snaps to converged w).
The reconciliation candidate from the ablation grid. Full protocols, both batteries, per-seed.
Bars: single mean>=0.85 min>=0.7 ovr<=0.10 | multi ho-exact>=0.75 swap<0.05 static>=0.9/seed, null@chance.
"""
import numpy as np
from collections import Counter
import neo_core
from neo_core import Brain, P

def cfg():
    P['turnover'] = 1; P['turnD2'] = 1; P['silentBirth'] = 1; P['sigFloor'] = 0.05
    P['anchor'] = 1; P['extinctAsym'] = 1; P['useGate'] = 1; P['crebGate'] = 1
    P['lamD1'] = 0.005; P['lamD2'] = 0.006; P['kCap'] = 0.08
OC = 1.6
SEEDS = (1, 3, 7, 11, 23, 42)

# ---------------- single battery ----------------
cfg()
print("=== VERDICT single: 12->345, 1400 trials, 20 asks ===", flush=True)
ex = {}; ovr = 0; n = 0
for s in SEEDS:
    b = Brain(s, nSym=8, nMem=160, overcomplete=OC)
    for _ in range(1400): b.train_trial([1, 2], [3, 4, 5])
    outs = Counter(''.join(map(str, b.ask([1, 2]))) for _ in range(20))
    ex[s] = outs.get('345', 0) / 20
    ov = sum(v for k, v in outs.items() if k.startswith('345') and len(k) > 3); ovr += ov; n += 20
    print(f"  seed {s:3d}: exact {ex[s]:.2f}  overruns {ov}/20  top={outs.most_common(3)}", flush=True)
print(f"  SINGLE: mean {np.mean(list(ex.values())):.2f}  min {min(ex.values()):.2f}  ovr {ovr/n:.2f}", flush=True)

# ---------------- multi battery ----------------
f = lambda x: x + 2
pairs = [(a, b) for a in range(4) for b in range(4) if a != b]
HO = [(0, 1), (1, 2), (2, 3), (3, 0)]
trainP = [p for p in pairs if p not in HO]
statics = [(6, [1]), (7, [0])]
def run_multi(s, nullMode, epochs=1000):
    cfg()
    rng = np.random.default_rng(s * 7919 + 13)
    b = Brain(s, nSym=8, nMem=160, overcomplete=OC)
    for ep in range(epochs):
        order = list(trainP); rng.shuffle(order)
        for (a, bb) in order:
            tgt = [int(rng.integers(0, 8)), int(rng.integers(0, 8))] if nullMode else [f(a), f(bb)]
            b.trial_reset(); b.encode_seq([a, bb])
            z = np.zeros(b.N)
            for _ in range(4): b.tick(z, False)
            b.emit_slots(tgt, 2, True)
            b.trials += 1
            if b.trials > P['warmupTrials']: b.prune_sweep()
        for (c, ans) in statics:
            b.trial_reset(); b.encode_seq([c])
            z = np.zeros(b.N)
            for _ in range(4): b.tick(z, False)
            b.emit_slots(ans, 1, True)
    def ev(ps, REP=6):
        pos = exx = sw = nn = 0
        for (a, bb) in ps:
            for _ in range(REP):
                em = b.ask([a, bb], max_slots=2); nn += 1
                if len(em) > 0 and em[0] == f(a): pos += 0.5
                if len(em) > 1 and em[1] == f(bb): pos += 0.5
                if em == [f(a), f(bb)]: exx += 1
                if em == [f(bb), f(a)]: sw += 1
        return pos / nn, exx / nn, sw / nn
    trPos, _, _ = ev(trainP)
    hoPos, hoEx, hoSw = ev(HO)
    sOk = sn = 0
    for (c, ans) in statics:
        for _ in range(8):
            em = b.ask([c], max_slots=1); sn += 1
            if em == ans: sOk += 1
    return trPos, hoPos, hoEx, hoSw, sOk / sn

print("=== VERDICT multi: compositional held-out + static, 1000 ep ===", flush=True)
res = []
for s in SEEDS:
    r = run_multi(s, False); res.append(r)
    print(f"  seed {s:3d}: trainPos {r[0]:.2f}  HO pos {r[1]:.2f} exact {r[2]:.2f} swap {r[3]:.2f}  static {r[4]:.2f}", flush=True)
m = np.mean(res, axis=0)
print(f"  MULTI: trainPos {m[0]:.2f}  ho-pos {m[1]:.2f}  ho-exact {m[2]:.2f}  swap {m[3]:.2f}  static {m[4]:.2f}", flush=True)
print("=== VERDICT null (3 seeds) ===", flush=True)
for s in (1, 7, 23):
    r = run_multi(s, True, epochs=600)
    print(f"  seed {s:3d}: trainPos {r[0]:.2f}  ho-pos {r[1]:.2f} exact {r[2]:.2f}", flush=True)
print("(bars: single .85/.70/.10 | multi ho-exact .75 swap .05 static .90/seed | null ~.125)")
