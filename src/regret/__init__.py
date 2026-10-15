"""Regret: an embeddable No-Limit Hold'em agent."""

from importlib.metadata import version as _dist_version
from typing import TYPE_CHECKING

from regret import _core

if TYPE_CHECKING:
    from regret.agent import RegretAgent

__version__ = _dist_version("regret-poker")

__all__ = ["RegretAgent", "__version__", "_core"]


def __getattr__(name: str) -> object:
    # Lazy, so `import regret` stays light for code that only needs the engine.
    if name == "RegretAgent":
        from regret.agent import RegretAgent

        return RegretAgent
    raise AttributeError(name)
