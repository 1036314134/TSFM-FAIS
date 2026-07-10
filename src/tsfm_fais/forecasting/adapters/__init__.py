"""Optional forecast model adapters."""

from .chronos import Chronos2Adapter, ChronosBoltAdapter
from .sundial import SundialAdapter
from .timesfm import TimesFM2p5Adapter
from .tirex import TiRexAdapter

__all__ = [
    "Chronos2Adapter",
    "ChronosBoltAdapter",
    "SundialAdapter",
    "TiRexAdapter",
    "TimesFM2p5Adapter",
]
