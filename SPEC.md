# SPEC: Regret — Embeddable Poker AI Agent (CFR + ML)

> Decisions that refine this spec (hardware sizing, latency targets, no Deep CFR from scratch, labels) are recorded in [`ROADMAP.md`](ROADMAP.md) §0 and §5.

## 1. Overview
A No-Limit Texas Hold'em agent that other systems embed and query for near-optimal decisions. Given a game state, it returns an action probability distribution and a recommended action. The core is Monte Carlo CFR (MCCFR) with abstractions and a compact neural approximation, so it stays small and fast enough to run on a normal CPU.

## 2. Goals
- Near-GTO play across all four streets (preflop, flop, turn, river).
- Small footprint: total model + blueprint **≤ 100 MB** (target ≤ 50 MB).
- Fast inference: **≤ 1s per decision on a laptop CPU**, including real-time re-solving on later streets.
- Clean, documented interface that other systems can use as a library or a service.
- Measurable strength (exploitability, win rate vs. baselines).

## 3. Scope
- *(Refined 2026-09-27: v1 ships heads-up only; a demo table ships as v1.1 and 6-max as v1.2. The engine stays 2–6-player from day one. See ROADMAP "Release scope".)*
- **v1: 2–6 player NLHE (heads-up through 6-max)**, fixed starting stack (e.g., 100bb, configurable), fixed blinds. Handle any number of players still in the hand on every street, including side pots and all-ins.
- Build and validate heads-up first internally as a checkpoint, but the architecture must be multiway from day one (no 2-player assumptions in the engine, state, or abstractions).
- **Later / stretch:** variable stack depths per player, 9-max, opponent modeling / exploitative adjustments.

### Multiway notes
- CFR carries no Nash equilibrium guarantee with 3+ players, but Pluribus showed MCCFR blueprints + depth-limited search still play at superhuman level in 6-max. Follow the Pluribus approach.
- The game tree grows fast with more players, so action abstraction must be coarser in multiway pots (fewer bet sizes when 3+ players are in the hand).

### Non-goals
- ~~No UI beyond a minimal CLI and demo script.~~ *(Refined 2026-09-28: a demo web table ships in v1.1; see ROADMAP M9.)*
- No integration with real-money poker sites.

## 4. Interface

### Input: GameState
```json
{
  "num_players": 6,
  "hero_seat": "BTN",
  "hero_cards": ["As", "Kd"],
  "board": ["Qh", "Jc", "2s"],
  "street": "flop",
  "pot": 20.5,
  "players": [
    {"seat": "UTG", "stack": 100.0, "status": "folded"},
    {"seat": "HJ",  "stack": 94.0,  "status": "active"},
    {"seat": "CO",  "stack": 100.0, "status": "folded"},
    {"seat": "BTN", "stack": 94.0,  "status": "active"},
    {"seat": "SB",  "stack": 100.0, "status": "folded"},
    {"seat": "BB",  "stack": 94.0,  "status": "active"}
  ],
  "action_history": {
    "preflop": [["UTG", "fold"], ["HJ", "raise", 2.5], ["CO", "fold"], ["BTN", "call", 2.5], ["SB", "fold"], ["BB", "call", 2.5]],
    "flop": [["BB", "check"], ["HJ", "bet", 6.0]]
  },
  "to_act": "BTN",
  "big_blind": 1.0
}
```
Bet amounts are in big blinds. `status` is `active`, `folded`, or `all_in`. Validate the input and return clear errors for illegal states. The engine must compute side pots correctly.

### Output
```json
{
  "strategy": {"fold": 0.10, "call": 0.55, "raise_2.5x": 0.25, "all_in": 0.10},
  "recommended": {"action": "call", "amount": 4.0},
  "source": "subgame_solve",
  "latency_ms": 640
}
```
`source` is `blueprint` or `subgame_solve`.

### Access modes
- Python library: `agent = RegretAgent.load(path); agent.act(state)`
- REST API (FastAPI): `POST /act`, `GET /health`, `GET /info`
- gRPC: optional, later phase.

## 5. Architecture

| Module | Responsibility |
|---|---|
| `engine/` | N-player (2–6) game rules, legal actions, state transitions, side pots, showdown with multiple players, hand evaluation (fast evaluator, lookup tables) |
| `abstraction/cards` | Hand bucketing per street: equity-distribution features, potential-aware clustering (k-means with EMD), more buckets preflop/flop, fewer on turn/river |
| `abstraction/actions` | Discrete bet sizes that depend on street and number of active players (e.g., 33%/75%/pot/all-in heads-up, only 50%/pot/all-in with 3+ players) + off-tree bet translation (pseudo-harmonic mapping) |
| `cfr/` | N-player MCCFR (external sampling, one traverser per iteration, rotating) with Linear CFR / discounting and negative-regret pruning; C++ or Numba for hot loops |
| `deepcfr/` | Deep CFR: advantage and strategy networks (PyTorch), reservoir-sampled memory buffers — *superseded: see ROADMAP D3 (`distill/`)* |
| `solver/` | Pluribus-style depth-limited search: at the leaf, each remaining player picks among a few continuation strategies (blueprint + biased toward fold/call/raise). Used on flop-onward, and earlier when the hand leaves the blueprint's abstraction |
| `agent/` | Public API: loads artifacts, routes to blueprint or solver, returns output |
| `api/` | FastAPI service, request validation, logging |
| `eval/` | Exploitability (local best response), head-to-head matches, mbb/hand with confidence intervals |
| `export/` | ONNX export of networks for portable inference |

## 6. Training pipeline
1. Precompute hand equity features and build bucket tables (cache to disk).
2. Train the blueprint with MCCFR on the abstracted game.
3. Train Deep CFR networks to replace or compress the tabular blueprint where it is large. *(Refined: distill the tabular blueprint into a policy net; see ROADMAP D3.)*
4. Export artifacts (bucket tables, networks as ONNX, config) into one versioned bundle.
5. Log training runs and checkpoint regularly (see 6.1).

### 6.1 Long-running training requirements
Training runs for hours to days in the background, on a laptop or a remote cloud server. It must:
- **Checkpoint automatically**: every N iterations and every M minutes (both configurable), save regrets/strategy sums, network weights, optimizer state, RNG seeds, and the iteration count. Write atomically (temp file then rename) and keep the last K checkpoints so a crash mid-write never corrupts progress.
- **Resume cleanly**: `python train.py --resume` picks up from the latest checkpoint with identical results to an uninterrupted run (seeded).
- **Shut down gracefully**: on Ctrl+C / SIGTERM, finish the current iteration, save a checkpoint, then exit.
- **Log progress**: iterations/sec, elapsed time, ETA, memory usage, loss, periodic exploitability (heads-up) and quick win rate vs. baseline bots. Write to a plain log file and TensorBoard.
- **Status command**: `python status.py` prints the latest progress summary from the log/checkpoint, so I can check in over SSH without attaching to the process.
- **Background-friendly**: include a `scripts/run_background.sh` that launches training with `nohup` (or inside `tmux`), redirects output to a log, and on macOS uses `caffeinate -i` to prevent sleep. Include short README instructions for running on a laptop and on a rented cloud server (setup, launch, detach, reattach, download checkpoints).
- **Usable mid-training**: any checkpoint can be exported into an agent bundle and evaluated while training continues.

Note: Pluribus's 6-max blueprint took ~8 days on a 64-core server. Size bucket counts and bet sizes so a usable 6-max blueprint trains in days on one machine, and document the tradeoffs.

Constraint: training must be feasible on a single machine (consumer GPU optional). Document expected compute time for each phase.

## 7. Evaluation
- **Exploitability** (heads-up only, where it is well-defined): abstract-game exploitability plus local best response (LBR) in the full game. For multiway, rely on head-to-head results below.
- **Head-to-head matches** in both heads-up and 6-max tables (duplicate/seat-rotated deals + AIVAT-style variance reduction) vs.:
  - random agent
  - always-call agent
  - simple rule-based agent (equity threshold)
  - earlier checkpoints of itself
- Report mbb/hand with 95% confidence intervals.
- Ablations: blueprint only vs. with subgame solving; bucket count vs. strength vs. size.

## 8. Engineering requirements
- Python 3.11+, PyTorch, NumPy, Numba and/or C++ (pybind11), FastAPI, pytest.
- Unit tests for engine rules, hand evaluation, action translation, and API validation.
- Deterministic mode (seeded) for reproducible evals.
- Config-driven (YAML) bet sizes, bucket counts, stack depth.
- README with quickstart, API docs, and architecture diagram.
- CI running tests + lint on push.

## 9. Success criteria (v1)
- Beats all baseline bots with statistical significance in both heads-up and 6-max.
- Subgame solving measurably improves win rate over blueprint-only.
- Meets size (≤ 100 MB) and latency (≤ 1s CPU) targets.
- Library and REST API both usable from a 10-line example script.

## 10. Open questions
Resolved — see ROADMAP §0 (D1–D8) and §5.
