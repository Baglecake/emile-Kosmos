# emile_mini - vendored core from https://github.com/Baglecake/emile-mini
# Only the QSE engine, config, and goal module needed by Kosmos.

__version__ = "0.6.0"

from .config import QSEConfig
from .agent import EmileAgent

__all__ = ["QSEConfig", "EmileAgent"]
