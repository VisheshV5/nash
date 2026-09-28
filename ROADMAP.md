# Regret — Roadmap

Status: **v0.4** (2026-09-28). Built from [`SPEC.md`](SPEC.md). All earlier open questions are resolved (§5). One new blocker, disk space, is listed in §4.

## Release scope

| Release | Scope | Calendar estimate |
|---|---|---|
| **v1: heads-up** | HU blueprint + real-time search + library + REST API, prebuilt wheels + auto-downloaded bundles, Slumbot benchmark | **~4 weeks** |
| v1.1: demo | Web poker table: play against it or watch bot vs. bot, with a panel showing the numbers behind each decision | ~1 week |
| v1.2: 6-max | 6-max blueprint (covers 3–6 players), multiway search, distillation | ~2 more weeks |
| Stretch | Learned best response (a stronger exploitability estimate than LBR) | ~1 week + compute |

**The engine, state, schema and abstractions stay 2–6-player from day one** (Spec §3), so v1.2 adds a training config, a blueprint and eval. It does not rewrite v1. Design notes for 6-max are kept below and marked *(v1.2)*.

## What makes Regret different

Open-source poker AI today is mostly research frameworks, and commercial solvers are heavy desktop apps. Regret's niche is **the poker AI you can drop into your own code in one minute**:

- `pip install regret-poker` with **prebuilt wheels** for macOS-arm64 and Linux x86_64. No compiler needed.
- `RegretAgent.load("hu-100bb")` **downloads the bundle automatically** from GitHub Releases on first use and caches it.
- One JSON format in and out, the same for the library and the REST API, and nothing to configure for the default.
- Published, measured footprint: bundle MB, load time, and p50/p95 decision latency on a laptop.
- **Honest strength numbers:** abstract-game exploitability, LBR, head-to-head vs. baselines, and **vs. Slumbot**, all with 95% CIs.
- **Every decision can explain itself** with numbers the agent actually computed (`explain=True`), and the demo table (v1.1) shows them next to each move.

## Target machine (the only machine; no cloud, no GPU)

| | |
|---|---|
| CPU | Apple M2 Pro: 6 performance + 4 efficiency cores |
| RAM | 16 GB → **training budget 10 GB** (rest for macOS + headroom) |
| GPU | none used for CFR; the Apple GPU (PyTorch MPS) is optional for distillation only |
| Disk | **~12 GB free today. Needs ≥ 40 GB free before M3** (see §4) |

Every size, bucket count and compute estimate below is chosen to fit this box.

---

## 0. Key design decisions

### D1. C++ vs. Numba → **C++17 + pybind11 for the hot path; Numba/NumPy only for offline batch work**
- MCCFR traversal is recursive, full of branches and keyed by lookups. Numba handles that poorly. C++ gives flat `int32` regret arrays, atomics for multithreaded updates, and one code path shared by training and real-time search.
- Numba or NumPy handles the embarrassingly parallel offline jobs: equity-histogram features and clustering.
- Python orchestrates: config, checkpoints, logging, eval, API.
- Build: `scikit-build-core` + CMake + pybind11. `uv sync` builds the extension.

### D2. Blueprints → **two: `hu` (v1) and `6max` (v1.2)**
- **Tables of 3–6 players use the 6-max blueprint through "virtual folds."** A 4-handed table is exactly the 6-max game after UTG and HJ fold: same blinds, same order, same stacks. CFR trains those subtrees directly because early-seat folds are the most common preflop action.
- **Heads-up is separate.** In HU the SB is the button and acts *last* postflop. In "6-max after 4 folds" the SB acts *first* postflop.

### D3. Blueprint representation → **tabular MCCFR; postflop distilled into a network only if needed for size; no Deep CFR from scratch**
- **v1 (HU):** ship the tabular blueprint if the bundle is ≤ 100 MB. Distill (M8) only if it isn't.
- **Preflop: tabular and lossless** (169 hand classes). The average strategy is stored as `uint8` probabilities.
- **Postflop: tabular during training, then distilled.** Postflop uses Pluribus-style snapshots of the current strategy (no postflop average-strategy accumulator, which saves about half the RAM). After training, a small MLP learns the tabular postflop strategy by supervised training and ships as ONNX. The tabular postflop is kept in the bundle only if it fits the size budget.
- Deep CFR from scratch is **out of scope**. Spec's `deepcfr/` module becomes `distill/`.

### D4. Size budget → **store cluster centroids, not lookup tables, for turn and river**
Card-isomorphic index counts, with the board treated as one set (hole + board): preflop 169 · flop 1,286,792 · turn 13,960,050 · river 123,156,254. (Keeping the turn and river cards separate would give 55,190,538 and 2,428,287,420; the abstraction doesn't need that distinction.) A `uint8` river table alone would be about 123 MB, so turn and river buckets are computed at runtime from centroids.

| Artifact | Strategy | Est. size |
|---|---|---|
| Preflop strategy, 6-max *(v1.2)* | tabular `uint8`, reachable sequences only, raise cap 4 | 10–20 MB |
| Preflop strategy, HU | tabular `uint8` | ≤ 1 MB |
| Flop buckets | `uint8` table over isomorphic index | 1.3 MB |
| Turn / river buckets | centroids + runtime features (≤ 5 ms / ≤ 1 ms) | < 1 MB |
| Postflop strategy, HU (v1) | tabular `uint8` if it fits, else MLP (ONNX fp16) | 5–60 MB, measured in M4 |
| Postflop policy net, 6-max *(v1.2)* | MLP, ONNX fp16 | 5–10 MB |
| Hand evaluator | bit-arithmetic evaluator, no lookup tables (not the 130 MB 2+2 table) | 0 MB |
| **Total** | | **≈ 20–40 MB** (target ≤ 50, hard cap 100) |

### D5. Action abstraction (all YAML)
| Situation | Bet/raise sizes |
|---|---|
| Preflop, open | 2.0–2.5bb (by position), all-in |
| Preflop, 3-bet / 4-bet | one size each (≈3x IP / 4x OOP, ≈2.3x), all-in; raise cap 4 |
| Postflop, 2 active (HU blueprint) | 33%, 75%, 125% pot, all-in; raises: 75% pot / all-in |
| Postflop, 2 active (inside 6-max blueprint) *(v1.2)* | 50%, pot, all-in; raises: pot / all-in |
| Postflop, 3 active | 50%, pot, all-in; raises: all-in |
| Postflop, 4+ active | pot, all-in |

Real-time search uses a **richer** action set than the blueprint and inserts the opponent's actual off-tree bet, so off-tree bets get solved rather than only translated.

### D6. Card abstraction: sized for a 10 GB RAM budget **[provisional, fixed by the memory planner in M3]**
| Street | HU (v1) | 6-max *(v1.2)* | Method |
|---|---|---|---|
| Preflop | 169 | 169 | lossless |
| Flop | 200 | 96 | potential-aware: EMD k-means over turn-bucket histograms |
| Turn | 200 | 64 | EMD k-means over river-equity histograms |
| River | 200 | 64 | OCHS (equity vs. 8 opponent-hand clusters), L2 k-means |

- The 6-max blueprint is deliberately coarse, roughly 50× less memory than Pluribus. **Real-time search makes up the difference.** Inside a subgame, hands are bucketed much more finely (≈ 500–1000 buckets from runtime features), so decisions on later streets are made at fine granularity even though the blueprint is coarse.
- Regret blocks are **allocated lazily** (only for sequences actually visited). With the pruning in D1, much of the theoretical tree is never allocated.
- A **memory planner** (`scripts/plan_memory.py`) computes the worst case and a sampled estimate of RAM from a config *before* training. It refuses to launch above the budget. Bucket counts are finalized with it.

### D7. Latency → **p95 ≤ 1.0 s, hard cap 1.5 s, on the M2 Pro**
- The solver is **anytime**. Its default wall-clock budget is 800 ms on 6 threads (the P-cores). It returns the best strategy so far. If it hasn't reached a minimum iteration count, it returns the blueprint with `source: "blueprint"`. The hard cap is enforced by the agent, not only the solver.
- Subgames are depth-limited to the end of the current street. Leaves use 4 continuation strategies per player. Opponent ranges are bucketed.
- **v1 (HU):** search runs on every flop, turn and river decision, and preflop when off-tree.
- **Routing policy (v1.2, configurable):** search runs when ≤ 3 players are in the pot, or when the state is off-tree. With 4+ players, the agent uses blueprint + translation. M10 measures whether 4-way search fits the budget.
- Blueprint-only decisions: ≤ 20 ms.

### D8. Training on a laptop
- Threads default to 6 (P-cores). E-cores are left for the OS and `status.py`. Configurable.
- `run_background.sh` uses `caffeinate -i` (plus `-s` when on AC) inside `tmux`. The README says to train plugged in and with the lid open or in clamshell mode with an external display.
- Checkpoints are zstd-compressed, keep `K=2` by default on this machine, and the directory is configurable (e.g. an external SSD).

---

## 1. Milestones

Estimates are **calendar time working with Claude Code most days**. Compute is wall-clock on the M2 Pro and overlaps with dev work. **All compute numbers are estimates until the throughput benchmark at the end of M2 replaces them.**

```
v1 (heads-up):
M0 ─► M1 ─► M2 ─► M3 ─► M4 (HU blueprint) ─► M5 (HU search) ─► M7 (bundle) ─► M8 (v1 release)
                          │                         │
                          └──► M6 (agent + API) ◄───┘   thin version right after M4

v1.1 (demo):   M9 (web table + decision panel)

v1.2 (6-max):  M10 (6-max blueprint + multiway search) ─► M11 (6-max distill + v1.2 release)
```

### M0 — Foundations (~1–2 days)  ✅ *done 2026-09-28*
**Deliverables**
- Repo scaffold (§2), `pyproject.toml` (scikit-build-core, pybind11), CMake, `uv` lockfile, MIT `LICENSE`.
- Tooling: ruff, mypy (strict on `agent/`, `api/`), clang-format, pre-commit.
- GitHub Actions CI (public repo → free minutes) on `macos-14` (arm64) and `ubuntu-latest`: build extension, pytest, lint, with ccache.
- Config system: YAML → pydantic (`configs/hu_default.yaml`; `6max_default.yaml` comes in v1.2).
- Seeding utility: one master seed → per-component NumPy `Generator` and C++ PCG streams.

**Acceptance:** CI is green on both runners. `uv sync && pytest` passes locally, and a trivial C++ function is callable from Python.
**Depends on:** nothing.

### M1 — N-player game engine + hand evaluation (~4–5 days)  ✅ *done 2026-09-28*
Built for 2–6 players even though v1 trains only heads-up.

**Deliverables**
- C++: cards, deck, 7-card evaluator (own bit-arithmetic implementation, no tables, nothing vendored), suit-isomorphism hand indexer for all streets (any round structure).  ✅ *done 2026-09-28: 62M evals/s on one M2 Pro core*
- C++ `GameState` for 2–6 seats: blinds, legal actions (min-raise rule; incomplete all-in raises don't reopen action), transitions, street advance, **side pots**, multiway showdown with split pots and odd chips.
- Python bindings and a pydantic `GameStateIn` matching Spec §4, with semantic validation: pot equals contributions, stacks are consistent with history, turn order is legal, no duplicate cards, board length matches the street, seat labels are valid for N. Errors are structured.

**Acceptance**
- Evaluator: exhaustive 5-card category counts over 2,598,960 hands are correct. ≥ 20M 7-card evals/s on one P-core.
- Engine: ≥ 1M random hands (2–6 players) **differentially fuzzed against PokerKit** with identical legal actions, pots, side pots and payouts. A fixtures file covers side-pot edge cases.
- Indexer: bijection verified, and counts match D4.
- The Spec §4 example parses. 20+ malformed states are rejected with specific messages.

**Depends on:** M0.

### M2 — MCCFR core on toy games + training infrastructure (~4–5 days)
**Deliverables**
- C++ N-player external-sampling MCCFR: traverser rotates each iteration, Linear CFR / discounting, negative-regret pruning (after warm-up, skip actions below the threshold 95% of the time), lazy regret allocation, multithreaded with atomic `int32` regrets.
- Generic game interface. The same trainer runs Kuhn, Leduc, 3-player Kuhn/Leduc, and (in M4) abstracted NLHE.
- Exact best response / exploitability for 2-player toy games.
- **Training harness (Spec §6.1), built once and reused:** `train.py`, checkpoints every N iters and every M minutes, atomic write (tmp → fsync → rename), keep last K, zstd, `--resume`, graceful stop on SIGINT/SIGTERM (finish the iteration, checkpoint, exit), log file + TensorBoard (it/s, elapsed, ETA, RSS, loss), `status.py`, `scripts/run_background.sh` (tmux + `caffeinate`).
- **Throughput benchmark** on the M2 Pro. Its output replaces the compute estimates in M4.

**Acceptance**
- Kuhn: exploitability < 1e-3, and game value within 1e-3 of −1/18. Leduc: exploitability < 0.01 chips/hand and falling on a log-log plot.
- 3-player Kuhn converges near published equilibria (sanity check that the core is N-player).
- **Resume determinism:** in deterministic mode (fixed thread count), 1000 iters followed by a kill and resume for 1000 more gives byte-identical regrets to 2000 uninterrupted iters. Kill -9 mid-checkpoint never corrupts the latest valid checkpoint (tested).

**Depends on:** M0 (M1 only for the NLHE adapter at the end).

### M3 — Card + action abstraction (~4–5 days, +~0.5 day compute)
**Deliverables**
- Features (C++/Numba): river OCHS, turn equity histograms, flop potential-aware histograms. **Disk-light design:** river and turn centroids are fit on a *sample* of isomorphic states (≈ 5–10M), not the full 123M. Features are float16 and streamed to disk. Flop features are computed in full (1.3M).
- Clustering: k-means (L2) and EMD k-means (fast 1-D EMD for histograms), deterministic seeding, and a bucket-quality report.
- Runtime bucketers: flop table lookup, turn/river centroid assignment, and a fine-grained mode for search (D6).
- Action abstraction per D5 (conditioned on active-player count, so v1.2 only needs config). Off-tree translation: randomized pseudo-harmonic mapping (seedable).
- `build_abstraction.py` → versioned `abstraction/` artifact. **`plan_memory.py`** (D6).

**Acceptance**
- HU abstraction build finishes in ≤ 12 h and uses ≤ 8 GB of scratch disk. Runtime turn bucketing ≤ 5 ms, river ≤ 1 ms.
- Pseudo-harmonic mapping matches a closed-form table, and probabilities sum to 1.
- Same seed gives an identical table hash. Each artifact's size is reported and fits D4.
- The memory planner's prediction for the HU config is within the 10 GB budget. HU bucket counts are finalized (with the whole 10 GB for HU, they may go above 200).

**Depends on:** M1.

### M4 — Heads-up blueprint + evaluation harness ★ (~4–5 days, +2–4 days compute)
**Deliverables**
- NLHE adapter for the MCCFR core. Preflop average strategy, postflop snapshots (D3).
- `export.py`: checkpoint → agent bundle. Works mid-training.
- `eval/`: baselines (random, always-call, equity-threshold rule bot), match runner (duplicate deals + seat rotation), AIVAT, mbb/hand ± 95% CI, HU abstract-game exploitability, **LBR** in the full game.
- Periodic in-training eval: exploitability and win rate vs. baselines, logged to TensorBoard.

**Acceptance**
- Beats all three baselines HU with the 95% CI excluding 0. Later checkpoints beat earlier ones.
- Abstract exploitability falls across checkpoints. The LBR number is recorded as the baseline for M5.
- Peak training RSS ≤ 10 GB. Blueprint-only decision latency ≤ 20 ms. Tabular bundle size measured (decides whether M7 distills).

**Depends on:** M2, M3.

### M5 — Depth-limited real-time search, heads-up (~5–7 days)
Search code is written for N players; v1 tests and tunes it heads-up only.

**Deliverables**
- Subgame construction from the public state, with beliefs from the blueprint (Bayes over action history). Pluribus-style search from the start of the current street.
- Leaf evaluation: each player picks among 4 continuation strategies (blueprint; blueprint biased toward fold / call / raise).
- Richer action set in search, with the actual off-tree bet inserted. Fine-grained hand buckets inside the subgame.
- Anytime solver, 6 threads, 800 ms budget, blueprint fallback (D7).

**Acceptance**
- **Blueprint + search beats blueprint-only** with 95% CI (AIVAT) over ≥ 200k duplicate hands. LBR improves over M4.
- p95 latency ≤ 1.0 s and max ≤ 1.5 s over 10k random HU decisions on the M2 Pro.

**Depends on:** M4. (Can be developed while the M4 blueprint is still training, using mid-training checkpoints.)

### M6 — Agent library + REST API (~2–3 days; thin version right after M4)
**Deliverables**
- `RegretAgent.load(path)` / `.act(state)`. Routes to blueprint, translation or solver. Returns the Spec §4 output.
- `recommended` is **sampled** from the strategy (seeded in deterministic mode), with optional `mode="argmax"`.
- `act(state, explain=True)` adds an `explain` object containing **only values the agent computed**: hero equity vs. the estimated range, hand bucket, a compact opponent-range summary (top hand classes + weights), per-action EV from the subgame solve (when `source` is `subgame_solve`), solver iterations, and the translation applied to an off-tree bet. No generated prose in the library.
- In v1, inputs with `num_players > 2` return a clear `UnsupportedTableSize` error. Schema validation already handles 2–6.
- FastAPI: `POST /act`, `GET /health`, `GET /info` (bundle version, config, abstraction hash), structured errors, request logging, Dockerfile.
- `examples/library.py`, `examples/rest.py` (≤ 10 lines each), and a CLI demo that plays a heads-up hand vs. baselines.

**Acceptance:** both examples run from a clean install. API validation tests pass. OpenAPI docs are generated.
**Depends on:** M4 (thin), M5 (full routing).

### M7 — Bundle, packaging, distribution (+ distillation only if needed) (~3–4 days; +3–4 days if distilling)
**Deliverables**
- **Prebuilt wheels** with `cibuildwheel` in a GitHub Actions release workflow (macOS-arm64, Linux x86_64 manylinux; Python 3.11–3.13), published to PyPI as `regret-poker` (import name `regret`).
- **Bundle registry:** bundles attached to GitHub Releases, a small `registry.json` (name → URL, sha256, size, version), `RegretAgent.load(name_or_path)` downloads, verifies the checksum and caches in `~/.cache/regret`. Offline use works with a local path.
- Lean dependencies: the core install needs only NumPy (+ ONNX Runtime if distilled). FastAPI, PyTorch and training tools are extras (`[api]`, `[train]`).
- Versioned bundle: `manifest.json` (version, config hash, sizes, iterations), abstraction, preflop tables, postflop strategy.
- **If the tabular HU bundle is > 100 MB (measured in M4):** distill postflop into an MLP. Inputs: bucket/equity features, pot and stack ratios, encoded action history. Loss is KL weighted by reach probability. Trained on CPU, or MPS if faster. ONNX fp16, ONNX Runtime in the agent.

**Acceptance**
- Bundle ≤ 100 MB (target ≤ 50). Load time ≤ 2 s.
- On a fresh macOS and a fresh Linux machine (CI), `pip install regret-poker` + the 10-line library example work with no compiler and no manual download.
- If distilled: distilled vs. tabular head-to-head loss ≤ 10 mbb/hand (CI reported).

**Depends on:** M4.

### M8 — Slumbot benchmark, ablations, docs, v1 release (~4–5 days, +2–4 days compute)
- **Slumbot match:** a client for Slumbot's public heads-up API. Before building it, **verify** Slumbot's current API, terms of use, stack depth and blinds (believed to be 200bb deep). If it's deeper than 100bb, train a matching `hu-200bb` bundle with the same pipeline (deeper stacks mean a bigger tree, so the memory planner re-checks the budget). Play ≥ 10k hands (as many as the API allows), rate-limited and politely, and report mbb/hand ± 95% CI whatever the result. Slumbot doesn't expose its strategy, so AIVAT can only use our own side; the report states that.
- Only bots with public APIs or open-source code. Never bots on real-money sites (Spec non-goal).
- Ablations (HU): blueprint vs. +search, bucket count vs. strength vs. size (3 points per street), distilled vs. tabular if M7 distilled.
- README (quickstart, API, architecture diagram, "training on your laptop"), `docs/compute.md` with the measured time per phase on the M2 Pro.
- Spec §9 verification for heads-up → `RESULTS.md` (including Slumbot and the footprint numbers). Tag `v1.0.0`, publish wheels and bundles.

---

### v1.1 — Demo table

### M9 — Web poker table with decision panel (~5–7 days)
The UI is **just another client of the REST API**, so it doubles as proof that embedding is easy.

**Deliverables**
- `web/`: Vite + TypeScript + Preact single-page app, built to static files and served by the FastAPI app (`regret serve --demo`). No other backend.
- **Play mode:** you vs. Regret, heads-up, with configurable stack depth, a seeded deck option, and a hand history download.
- **Watch mode:** Regret vs. a baseline bot, or vs. an older checkpoint, with playback speed controls and a running mbb/hand chart.
- **Decision panel** next to each move, built only from the `explain` output: action probabilities as a bar chart, equity, an estimated-range grid (13×13), per-action EVs, blueprint vs. solver, solver iterations, and latency.
- Optional one-line text summary produced by a **deterministic template** from those numbers (e.g. "Calls 55%: 38% equity vs. a range weighted toward top pairs; raising has lower EV"). No LLM needed, and it can never contradict the numbers.
- Deployable as a single Docker image. Hosting publicly is optional and later.

**Acceptance**
- A full hand can be played in the browser against the v1 bundle on the M2 Pro with ≤ 1.5 s bot response.
- Every number in the panel matches the API response for the same state (tested).
- Watch mode runs 1,000 hands unattended without errors.

**Depends on:** v1 (M6 `explain`, M7 bundle).

---

### v1.2 — 6-max

### M10 — 6-max blueprint + multiway search (~1–1.5 weeks, +5–10 days compute)
**Deliverables**
- `configs/6max_default.yaml` (D5/D6 coarse settings). Memory plan within 10 GB. Training run with the M2 harness. **The laptop is heavily loaded for the whole run.**
- Virtual-fold mapping for 3–5-handed inputs (D2). Lift the `num_players > 2` restriction in the agent.
- Multiway search per the D7 routing policy. Latency profiled by players-in-pot.
- 6-max eval: seat-rotated duplicate tables (agent vs. 5 baselines, agent ×k vs. baselines), and vs. earlier checkpoints.

**Acceptance**
- Beats every baseline at 6-max with significance. Also measured 3- and 4-handed through virtual folds.
- Search beats blueprint-only in the pots where D7 routes to search. The final routing threshold (3-way vs. 4-way) is set from measured latency.
- Peak RSS ≤ 10 GB.

**Depends on:** v1.

### M11 — 6-max distillation + v1.2 release (~3–5 days)
- Distill the 6-max postflop into an MLP (needed: the 6-max tabular postflop won't fit the budget). Bundle holds both blueprints and stays ≤ 100 MB.
- 6-max ablations, docs, and Spec §9 verification for 6-max. Watch mode in the demo gains 6-max tables. Tag `v1.2.0`.

---

### Stretch — Learned best response (~1 week + 2–4 days compute)
- Train an exploiter against the frozen HU agent (tabular MCCFR best-response over the abstract game or an RL agent in the full game) and report how much it wins. This is a stronger lower bound on exploitability than LBR, and it's the most credible exploitability number we can produce on one machine.

### Timeline summary
| | Calendar | Compute on M2 Pro (est.) |
|---|---|---|
| M0–M3 (foundations, engine, CFR, abstraction) | ~2 weeks | ~0.5 day |
| M4–M5 (HU blueprint + search) | ~1.5 weeks | 2–4 days, overlapping M5 |
| M6–M8 (agent, API, packaging, Slumbot, release) | ~1.5 weeks | 2–4 days (200bb bundle, if needed) |
| **v1 (heads-up) total** | **~4 weeks** | |
| M9 (v1.1, demo table) | ~1 week | — |
| M10–M11 (v1.2, 6-max) | ~2 weeks | 5–10 days |

Out of scope for v1–v1.2 (Spec "later"): variable per-player stacks, 9-max, opponent modeling, gRPC.

---

## 2. Repo structure

Repo: `github.com/VisheshV5/nash`. Python import name: `regret`.

```
nash/
├── pyproject.toml            # scikit-build-core + pybind11; extras: [train], [api], [dev]
├── CMakeLists.txt
├── LICENSE / NOTICE
├── SPEC.md / ROADMAP.md
├── configs/                  # hu_default.yaml (v1), 6max_default.yaml (v1.2), eval_*.yaml
├── cpp/
│   ├── include/regret/
│   ├── src/
│   │   ├── cards/            # card, deck, evaluator, isomorphism indexer
│   │   ├── engine/           # N-player state, side pots, showdown
│   │   ├── abstraction/      # runtime bucketers, action abstraction
│   │   ├── cfr/              # MCCFR, regret storage, toy games
│   │   └── solver/           # depth-limited search
│   ├── bindings/             # pybind11 → regret._core
│   └── tests/                # Catch2 for C++-only invariants
├── src/regret/
│   ├── engine/               # wrappers, GameState schema & validation
│   ├── abstraction/          # cards/ (features, clustering), actions/ (translation)
│   ├── cfr/                  # training orchestration, checkpointing
│   ├── distill/              # PyTorch policy nets distilled from the blueprint
│   ├── solver/               # search config, routing policy
│   ├── agent/                # RegretAgent, bundle loader
│   ├── api/                  # FastAPI app
│   ├── eval/                 # baselines, match runner, AIVAT, LBR, exploitability
│   ├── export/               # bundle + ONNX export
│   └── utils/                # config, seeding, logging, atomic io
├── scripts/                  # train.py, status.py, build_abstraction.py, plan_memory.py, export.py, run_background.sh
├── tests/                    # pytest: unit, differential vs PokerKit, API
├── web/                      # demo table (v1.1): Vite + TypeScript + Preact
├── examples/
├── docs/
└── .github/workflows/ci.yml
```

---

## 3. First three tasks

1. **M0 scaffold + build pipeline.** Layout above, scikit-build-core/pybind11/CMake, stub `regret._core.version()`, YAML → pydantic config with both default configs, seeding utility, ruff/mypy/clang-format/pre-commit, CI on `macos-14` + `ubuntu-latest`, MIT license.
   *Done when:* CI is green, and `uv run python -c "import regret; print(regret._core.version())"` works.
2. **Cards, evaluator, isomorphism (C++ + bindings).**
   *Done when:* 5-card category counts are exact, indexer counts match D4 with a verified bijection, ≥ 20M evals/s on a P-core, and bindings are tested.
3. **N-player engine + GameState schema + PokerKit fuzzer.**
   *Done when:* 1M fuzzed hands match PokerKit, side-pot fixtures pass, the Spec §4 example parses, and malformed states fail with specific errors.

Then M2 (toy CFR + harness) and M3 (abstraction) can run in parallel.

---

## 4. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Slumbot API changes, rate limits, or terms forbid automated play | No external benchmark | Verify first in M8; fall back to open-source bots and publish our baselines |
| Laptop-trained blueprint loses to Slumbot | Weaker headline | Report honestly with CIs; the search ablation and footprint numbers still stand |
| **Only ~12 GB of free disk** | Blocks M3 (feature caches) and M4/M10 (checkpoints of several GB each) | **Free ≥ 40 GB before M3** (toolchain + PyTorch ≈ 3 GB, features ≤ 8 GB, 2 compressed checkpoints ≤ 15 GB, bundles/logs). Checkpoint dir can live on an external SSD |
| 16 GB RAM caps blueprint granularity, 6-max most of all | Weaker 6-max blueprint (v1.2) | Coarse D6 + lazy allocation + snapshot postflop; memory planner; real-time search with fine buckets does the heavy lifting |
| 6-max search can't fit 1 s at 4+ players (v1.2) | Search only helps ≤ 3-way pots | D7 routing policy; anytime solver; measured in M10 |
| Multi-day laptop training (sleep, heat, updates) | Lost runs | caffeinate + tmux, checkpoints every ≤ 15 min, bit-exact resume, pause macOS auto-updates during runs |
| Engine rule bugs (side pots, reopen rules) | Poisons all training | PokerKit differential fuzzing before any training |
| Bit-identical resume with multithreading | Spec §6.1 claim | Guaranteed in deterministic mode (fixed threads, deterministic reduction); documented |
| Distilled net loses strength | Size vs. strength | Keep flop tabular if it fits; bigger net; M8/M11 ablations |

---

## 5. Decisions log (2026-09-27)

| # | Question | Decision |
|---|---|---|
| 1 | Hardware | M2 Pro (6P+4E), 16 GB, no GPU, no cloud → 10 GB training budget, D6 sizes |
| 2 | Latency | p95 ≤ 1.0 s, hard cap 1.5 s, 6 threads, blueprint fallback (D7) |
| 3 | Deep CFR | Not from scratch. Tabular MCCFR blueprint, postflop distilled to a net (D3) |
| 4 | Stack depth | Training-time config. At runtime, unequal stacks → effective stack + warning |
| 5 | Antes / straddles | None in v1 |
| 6 | `recommended` | Sampled from strategy (seedable), `mode="argmax"` optional |
| 7 | Labels | Seats named back from the button: HU `BTN`(=SB), `BB`; 3 `BTN,SB,BB`; 4 adds `CO`; 5 adds `HJ`; 6 adds `UTG`. Actions `fold`, `check`, `call`, `bet_0.75pot`, `raise_2.5x` (multiple of the facing bet), `all_in` |
| 8 | Repo | Fresh repo `VisheshV5/nash` at `~/nash`, no code shared with `~/pokerai`. MIT license. Package name `regret` |
| 9 | Release scope | **v1 = heads-up only** (~3–4 weeks). 6-max moves to v1.2. Engine and abstractions stay 2–6-player |
| 10 | Reviewer feedback (2026-09-28) | Adopted: embeddability as the headline (wheels, auto-download bundles, `explain`); Slumbot benchmark in v1; demo table as v1.1, before 6-max, which becomes v1.2; learned best response as stretch |
