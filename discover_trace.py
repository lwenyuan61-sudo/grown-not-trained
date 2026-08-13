"""Traced discover life (seed 101): poor-only plenty, then winter + rich appears.
Output: discover_trace.jsonl (t, choice, E) for Fig. 6 left panel."""
import json
from neo_core import Brain, P
import discover_confirm as D

if __name__ == "__main__":
    P["needs"] = 1; P["hRace"] = 1; P["tauMetab"] = 500.0
    b = Brain(101)
    out = open("discover_trace.jsonl", "w", encoding="utf-8")
    for t in range(1800):
        if t < 1200:
            c, E = D.trial(b, 0.0, 0.08)
        else:
            c, E = D.trial(b, 0.25, 0.03)
        out.write(json.dumps(dict(t=t, choice=int(c), E=round(E, 3))) + "\n")
    out.close()
    print("done")
