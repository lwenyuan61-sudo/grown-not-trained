"""Full regression battery on the WINNING structural-plasticity config (SPEC 1.12).
The 12->345 task passed (AB7b: 0.96/0.80/0.04). Before the config becomes default, verify the
new machinery didn't break what mattered most:
  R1 compositional held-out : input (a,b) -> (f(a),f(b)), f=x+2; train 8 pairs, hold out a 4-cycle
                              whose reverses are trained. Bars: held-out exact >= 0.7, swap < 0.05.
  R2 shuffled-null          : target re-randomized every epoch -> held-out must sit at chance.
  R3 static interference    : 2 static cards trained interleaved -> >= 0.9 every seed.
"""
import numpy as np
import neo_core
from neo_core import Brain, P

P['turnover'] = 1; P['turnD2'] = 1; P['silentBirth'] = 1; P['sigFloor'] = 0.05
P['anchor'] = 1; P['lamD1'] = 0.005; P['lamD2'] = 0.006; P['kCap'] = 0.08

f = lambda x: x + 2
pairs = [(a, b) for a in range(4) for b in range(4) if a != b]
HO = [(0, 1), (1, 2), (2, 3), (3, 0)]
trainP = [p for p in pairs if p not in HO]
statics = [(6, [1]), (7, [0])]
SEEDS = (1, 3, 7, 11, 23, 42); EPOCHS = 1000; REP = 6

def run_seed(s, nullMode):
    rng = np.random.default_rng(s * 7919 + 13)
    b = Brain(s, nSym=8, nMem=160, overcomplete=1.6)
    for ep in range(EPOCHS):
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
    def ev(ps):
        pos = ex = sw = n = 0
        for (a, bb) in ps:
            for _ in range(REP):
                em = b.ask([a, bb], max_slots=2); n += 1
                if len(em) > 0 and em[0] == f(a): pos += 0.5
                if len(em) > 1 and em[1] == f(bb): pos += 0.5
                if em == [f(a), f(bb)]: ex += 1
                if em == [f(bb), f(a)]: sw += 1
        return pos / n, ex / n, sw / n
    trPos, trEx, _ = ev(trainP)
    hoPos, hoEx, hoSw = ev(HO)
    sOk = sn = 0
    for (c, ans) in statics:
        for _ in range(8):
            em = b.ask([c], max_slots=1); sn += 1
            if em == ans: sOk += 1
    return trPos, hoPos, hoEx, hoSw, sOk / sn

print("=== R1+R3: compositional (a,b)->(f(a),f(b)) held-out + static interference ===", flush=True)
res = []
for s in SEEDS:
    r = run_seed(s, False); res.append(r)
    print(f"  seed {s:3d}: trainPos {r[0]:.2f}  HELD-OUT pos {r[1]:.2f} exact {r[2]:.2f} swap {r[3]:.2f}  static {r[4]:.2f}", flush=True)
m = np.mean(res, axis=0)
print(f"  mean: trainPos {m[0]:.2f}  ho-pos {m[1]:.2f}  ho-exact {m[2]:.2f}  swap {m[3]:.2f}  static {m[4]:.2f}", flush=True)
print("=== R2: shuffled-null (target re-randomized every epoch) ===", flush=True)
nres = []
for s in (1, 7, 23):
    r = run_seed(s, True); nres.append(r)
    print(f"  seed {s:3d}: trainPos {r[0]:.2f}  ho-pos {r[1]:.2f} exact {r[2]:.2f}", flush=True)
nm = np.mean(nres, axis=0)
print(f"  null mean: trainPos {nm[0]:.2f}  ho-pos {nm[1]:.2f}  (pos chance 0.125)")
print("(bars: ho-exact>=0.7 swap<0.05 static>=0.9-every-seed null-at-chance)")
