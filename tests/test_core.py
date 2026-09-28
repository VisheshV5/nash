import regret
from regret import _core


def test_extension_loads_and_reports_package_version() -> None:
    assert _core.version() == regret.__version__
