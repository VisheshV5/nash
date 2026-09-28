# Regret

An embeddable No-Limit Texas Hold'em agent: an MCCFR blueprint plus real-time search, small enough
(≤ 100 MB) and fast enough (≤ 1 s per decision on a laptop CPU) to drop into your own code.

> **Status: pre-alpha.** Foundations (M0) only; nothing plays poker yet.
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

## Layout

| Path | What |
|---|---|
| `cpp/` | Native core (engine, abstraction, CFR, search) exposed to Python as `regret._core` |
| `src/regret/` | Python package: config, orchestration, agent, API, eval |
| `configs/` | YAML configs (`hu_default.yaml`) |
| `tests/` | pytest suite |

## License

MIT
