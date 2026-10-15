# Compute on a laptop

Every number here was measured on the project's only machine: an Apple M2 Pro (6 performance +
4 efficiency cores, 16 GB RAM), no GPU, no cloud. Training ran only on AC power
(`scripts/train_when_plugged.sh`). Measurements taken while training was running say so: the
trainer holds the six performance cores, so they are pessimistic.

## Pipeline, heads-up 100bb (`configs/hu_default.yaml`)

| Phase | Time | Notes |
|---|---|---|
| Card abstraction, 500 buckets per street | 28 min (1,673 s) | 6 threads. River fit 453 s, turn fit 455 s, flop fit 639 s; tables 126 s. |
| Blueprint training, first 1B iterations | ~3.5 h | ~80k iterations/s on 6 threads, linear CFR discounting. |
| Blueprint training, after 1B iterations | ~0.5–0.6M iterations/s | Negative-regret pruning on 95% of iterations: a 5.8x speed-up. |
| Blueprint size in memory | 24.6M infosets, 1.05 GB regret table, 1.33 GB RSS | Option D in ROADMAP M3. |
| Export to a bundle | not timed | 43 MB bundle (dense uint8 strategy + centroids). |
| Bundle load | 0.37 s | `RegretAgent.load` from a local directory. |

## Evaluation (one core each, while training ran)

| Run | Time |
|---|---|
| 6,000 duplicate hands vs random / always-call | 1–4 s |
| 6,000 duplicate hands vs the equity-threshold bot | ~75 s (equity sampling per decision) |
| 10,000 hands blueprint vs blueprint | ~8 s |
| 3,000 hands LBR | ~150–160 s |
| 2,000 hands blueprint + turn/river search vs blueprint | 27 min (1,915 searches at ~0.85 s) |

## Decision latency (3.03B-iteration bundle, while training ran)

Blueprint only, 2,889 decisions against the equity-threshold bot:

| Street | p50 | p95 | max |
|---|---|---|---|
| preflop | 0.1 ms | 0.2 ms | 1.2 ms |
| flop | 0.2 ms | 0.6 ms | 1.4 ms |
| turn | 0.6 ms | 1.4 ms | 2.4 ms |
| river | 0.2 ms | 0.5 ms | 1.2 ms |

With river search (800 ms budget, 6 threads), 200 river decisions: p50 837 ms, p95 849 ms,
max 861 ms. Targets (ROADMAP D7): p95 ≤ 1.0 s, hard cap 1.5 s. The turn and the other streets
are unchanged (blueprint).

Turn search (opt-in, `search.streets: [turn, river]`): about 3,200 tree nodes and 50 ms per
solver iteration on one core; about 80 iterations fit in the 800 ms budget on six idle cores.

## What doesn't fit

- 6-max with the heads-up abstraction needs far more than 16 GB (ROADMAP v1.2): 16-bit
  regrets, much coarser buckets, or a rented machine.
- A 200bb heads-up blueprint (for Slumbot) fits: the planner puts it at 52.5M infosets,
  2.4 GB (`configs/hu_200bb.yaml`).
