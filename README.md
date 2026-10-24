# Grown, not trained

Code, data, and live demonstrations for the paper **"Grown, not trained: a
backpropagation-free neural organism that learns while it lives, taught only by its own
body"** (Wenyuan Li, submitted).

Follow-up work — a continuously running brain-inspired recurrent network with one-shot
hippocampal memory — is live at **[bowl.chat](https://bowl.chat)**.

A recurrent neural tissue whose only learning mechanism is local three-factor chemistry —
no gradients, no objective function, no optimizer, no training phase. Coupled to a
metabolic body through three fixed wires (gut→dopamine scaled by need; damage→opponent
signal scaled by the insult; hunger-pang→opponent signal against the ongoing action), it
learns to feed itself, prefer richer food, rest when sated, refuse sweetened poison, and
survive famine — while a wire-severed twin with the identical brain and body learns none
of it.

## Quick start

Requirements: Python ≥ 3.12, NumPy ≥ 2.0. Nothing else.

```bash
python run_all.py        # 3-minute demo: one fully logged life
python run_all.py full   # reproduce every number in the paper (hours)
```

On Windows, double-click `run_all.bat`.

## Live in your browser — no install

Open either page in any modern browser (offline works):

- **`wanning_value_machine.html`** — the organism lives in a world of cards you define
  (name, energy, supply). Watch it learn what to want; ask it at any moment what it
  would choose; sever its teaching wires to meet the twin from the paper.
- **`wanning_learning_machine.html`** — teach it any symbolic pattern with example
  cards; press Start and it develops continuously (learning, pruning, anchoring);
  quiz it on pairs it was never taught.

Both pages draw every one of the 442 neurons and 24,926 synapses of the exported engine
connectome and run the paper's tick() physics live.

## Map of the repository

| File | What it is |
|---|---|
| `neo_core.py` | the engine — all physics |
| `value_confirm3.py` | Part II main pack: 30 seeds, always-on plasticity, severed twins |
| `c1_alwayson.py`, `swap_confirm.py`, `discover_confirm.py` | acquisition, food-follows, famine-rediscovery controls |
| `q_baseline.py` | tabular Q-learner given the identical payout (baseline) |
| `verdict_creb.py` | Part I substrate battery |
| `neo_regression.py` | bit-exact golden regressions |
| `trace_life.py`, `discover_trace.py` | fully logged single lives (figure data) |
| `*.jsonl` | raw per-life records behind every number |
| `CELL_ATLAS.md` | cell-by-cell atlas: biology ↔ code |
| `README_REPRO.md` | claim → command → output table |

License: Apache-2.0.
