# Regret

An embeddable No-Limit Texas Hold'em agent: an MCCFR blueprint plus real-time search, small enough
(≤ 100 MB) and fast enough (≤ 1 s per decision on a laptop CPU) to drop into your own code.

> **Status: pre-alpha.** The rules engine is done (cards, evaluator, hand indexer, 2-6 player betting with side pots, GameState validation); no strategy yet.
> See [ROADMAP.md](ROADMAP.md) for the plan and [SPEC.md](SPEC.md) for the requirements.

## Development

Requires [uv](https://docs.astral.sh/uv/) and a C++17 compiler (Xcode Command Line Tools on macOS).
CMake and Ninja are fetched automatically during the build.

```bash
uv sync                # create .venv, build the C++ extension, install dev tools
uv run pytest          # tests
uv run ruff check .    # lint
uv run mypy            # type-check
uv run pre-commit install
```

Editing files under `cpp/` triggers a rebuild on the next `uv sync` or `uv run`.

Exhaustive C++ checks (every 5- and 7-card hand, every flop deal, index round trips) plus an
evaluator benchmark live in a separate binary:

```bash
uv run cmake -S . -B build/cpp-tests -G Ninja -DREGRET_BUILD_TESTS=ON
uv run cmake --build build/cpp-tests
./build/cpp-tests/regret_tests          # ~8 s
./build/cpp-tests/regret_tests --slow   # + every turn deal, every turn/river index (~3 min)
```

## Layout

| Path | What |
|---|---|
| `cpp/` | Native core (engine, abstraction, CFR, search) exposed to Python as `regret._core` |
| `src/regret/` | Python package: config, orchestration, agent, API, eval |
| `configs/` | YAML configs (`hu_default.yaml`) |
| `tests/` | pytest suite |

## License

MIT
