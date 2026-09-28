"""Regret: an embeddable No-Limit Hold'em agent."""

from importlib.metadata import version as _dist_version

from regret import _core

__version__ = _dist_version("regret-poker")

__all__ = ["__version__", "_core"]
