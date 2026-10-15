# Results

Measured on an M2 Pro laptop (6 performance cores, 16 GB). mbb/hand = thousandths of a big blind
won per hand. Confidence intervals are 95%. Matches are duplicate (same cards, seats swapped) with
an all-in luck adjustment. These numbers are updated as the v1 blueprint trains.

## Correctness

| Check | Result |
|---|---|
| Hand evaluator, every 5- and 7-card hand | category counts exact (133,784,560 seven-card hands); 62M hands/s on one core |
| Betting engine vs PokerKit | 1,200,000 random 2-6 player hands, 8.5M decisions, 0 differences |
| Suit-isomorphic hand indexing | flop/turn/river class counts exact (1,286,792 / 13,960,050 / 123,156,254) |
| MCCFR on Kuhn poker | exploitability 0.0003; first player's value -1/18 |
| MCCFR on Leduc hold'em | exploitability 0.0024 after 30M iterations |
| River search vs theory (polarized bettor vs bluff catcher) | nuts always bet; bluffs:value 1:2; catcher calls 1/2 vs pot and 1/3 vs 2x pot |

## Heads-up 100bb blueprint (training in progress)

24.6M information sets, 500 buckets per postflop street. Trained on 6 threads at ~80k
iterations/s. Bundle: 45 MB.

**After 1 hour (314M iterations)**, 6,000 hands per baseline:

| Opponent | mbb/hand |
|---|---|
| random | +1,611 ± 515 |
| always-call | +1,792 ± 315 |
| equity-threshold rule bot | +860 ± 366 |

All three are significant wins. Head-to-head against the 167M-iteration checkpoint: -15 ± 235
(no difference yet, as expected this early). LBR exploitability lower bound: 1,313 ± 1,169
mbb/hand (600 hands; the restricted LBR described in ROADMAP M4).

**Blueprint + river search vs blueprint alone** (1-hour blueprint, 300 ms per search while
training occupied the performance cores, so about 64 solver iterations per decision): +57 ± 206
mbb/hand over 8,000 hands (3,372 searches). Not significant. A first 1,000-hand run gave
+209 ± 526. The solver's answers at that budget are within ~2.7% of the pot of the subgame's
equilibrium. To re-measure on the mature blueprint, with the full budget and no training running.

## Still to measure

- The blueprint after the full run, and LBR with larger samples.
- River search on the mature blueprint.
- Slumbot (needs a 200bb blueprint; Slumbot plays 200bb stacks).
