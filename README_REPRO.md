# Reproduction guide — "Grown, not trained" (manuscript v4)

Every quantitative claim in the paper maps to one command below. All randomness flows
from the seed lists printed inside each script; there is no optimizer anywhere in the
pipeline. Confirmatory seeds are the first primes > 100, fixed before the confirmatory
runs; exploratory development used different seeds and is not part of any claim.

## Environment
- Python ≥ 3.12, NumPy ≥ 2.0; matplotlib only for figures.
- Deterministic per machine; golden regressions are bit-exact under a fixed environment.

## One click
`run_all.py` (or double-click `run_all.bat` on Windows): quick demo ≈ 3 minutes
(one fully-logged life); `python run_all.py full` reproduces every number (hours).

## Claim → command → output

| Paper claim | Command | Output |
|---|---|---|
| Part I substrate battery: held-out 1.00 / swapped 0.00 / static 1.00 / null ≈ chance (6 seeds) | `python verdict_creb.py` | console per-seed |
| Physics unchanged across additions (bit-exact under fixed env) | `python neo_regression.py` | golden transcripts |
| Part II main pack, 30 seeds, always-on plasticity, severed twins: preference 26/30, state-modulation 28/30, poison 29/30, homeostasis-vs-twin 29/30, winter 29/30 | `python value_confirm3.py` | value_confirm3.jsonl |
| Acquisition from chance, body vs sham wire (10 seeds, 10/10) | `python c1_alwayson.py` | c1_ao.jsonl |
| Location swap: preference follows the food (10/10, 0.98–1.00) | `python swap_confirm.py` | swap_ao.jsonl |
| Famine rediscovery (9/10) | `python discover_confirm.py` | discover_confirm.jsonl |
| Tabular Q baseline, identical payout/observations/budget (30 seeds) | `python q_baseline.py` | q_baseline.jsonl |
| Fig. 2 fully-logged life | `python trace_life.py` | life_trace.jsonl |
| Fig. 6 logged discovery life | `python discover_trace.py` | discover_trace.jsonl |
| Figures 1–6 | `cd paper/figs && python build_figs3.py` | fig*.png/pdf |
| English submission draft | `cd paper && python split_en.py` | manuscript_EN.md |

## Notes
- The main pack runs with plasticity active at every tick (`alwaysOn=1`, `datPump=1`);
  the `learn` argument in rig code carries teaching conductances only.
- The severed twin is the identical brain/body/protocol with the three teaching
  conductances set to zero; its plasticity, decay, and turnover run identically.
- Constants and their provenance: Methods Table 1 and `CELL_ATLAS.md` §10.
- Historical materials from earlier development phases (superseded packs, retired
  protocols) remain in the repository history but support no claim in the paper.
- Browser demonstrations are illustrations, not evidence, and are labeled as such.
