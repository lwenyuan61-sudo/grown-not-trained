"""Tabular RL baseline (Reviewer 1, M2): epsilon-greedy Q-learner given the SAME
scalar reward the wires deliver, the same 8 actions, the same encounter budget, and
the same interoceptive observation (hunger bin). Honest comparison, whatever it shows.
State: hunger level binned to 3 (hungry/mid/full). Actions: 8 (4 mines + 4 wander).
Reward per encounter: kb*dE - kh*damage - kp*h*(h>0.5), identical to the wires' payout.
Same worlds: plenty 1200 (rich .25/poor .08) then winter 600 (poor .03); body physics
identical (burn 0.05/encounter, E in [0.05,1]). 30 seeds, same primes.
"""
import json
import numpy as np

SEEDS = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
         179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257]
RICH, POOR, EMPTY, TOX = 0, 1, 2, 3
KB, KH, KP = 3.2, 3.2, 0.6
ALPHA, GAMMA, EPS = 0.2, 0.9, 0.1

def hbin(E):
    d = max(0.7 - E, 0.0)
    h = np.tanh(4 * d)
    return 0 if h >= 0.3 else (2 if h <= 0.05 else 1)

def run(seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 8))
    E = 0.7
    log = []
    for t in range(1800):
        poor_meal = 0.08 if t < 1200 else 0.03
        s = hbin(E)
        a = rng.integers(0, 8) if rng.random() < EPS else int(np.argmax(Q[s]))
        dE = harm = 0.0
        if a == RICH:
            dE = min(0.25, 1 - E); E += dE
        elif a == POOR:
            dE = min(poor_meal, 1 - E); E += dE
        elif a == TOX:
            dS = min(0.05, 1 - E); E += dS
            E = max(0.05, E - 0.25)
            dE, harm = dS, 0.25
        h = np.tanh(4 * max(0.7 - E, 0.0))
        r = KB * dE - KH * harm - (KP * h if h > 0.5 else 0.0)
        E = max(0.05, E - 0.05)
        s2 = hbin(E)
        Q[s, a] += ALPHA * (r + GAMMA * Q[s2].max() - Q[s, a])
        log.append((a, E))
    def stats(a0, z):
        seg = log[a0:z]
        mines = [a for a, _ in seg if a in (RICH, POOR, EMPTY, TOX)]
        pr = sum(1 for a in mines if a == RICH) / max(len(mines), 1)
        Es = np.array([e for _, e in seg])
        return round(pr, 2), round(float(Es.mean()), 2)
    pr1, e1 = stats(900, 1200)
    prw, ew = stats(1500, 1800)
    tox = sum(1 for a, _ in log if a == TOX)
    # acquisition speed: first block of 50 with P(rich among all acts) >= 0.5
    acq = next((i for i in range(0, 1750, 50)
                if sum(1 for a, _ in log[i:i+50] if a == RICH) / 50 >= 0.5), -1)
    return dict(seed=seed, P_rich_plenty=pr1, meanE_plenty=e1, toxic_total=tox,
                winter_P_rich=prw, winter_meanE=ew, acquisition_encounter=acq)

if __name__ == "__main__":
    rows = [run(s) for s in SEEDS]
    with open("q_baseline.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    def med(k): return float(np.median([r[k] for r in rows]))
    print(json.dumps(dict(median_P_rich=med("P_rich_plenty"), median_E=med("meanE_plenty"),
                          median_tox=med("toxic_total"), median_acq=med("acquisition_encounter"),
                          median_winter_P_rich=med("winter_P_rich"))), flush=True)
