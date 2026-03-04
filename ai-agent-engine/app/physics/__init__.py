"""
Travion Physics Engine Package.

This package contains research-grade physics calculations for:
- Solar position and golden hour computation
- Atmospheric modeling
- Topographic corrections

All calculations use the SAMP (Solar Azimuth and Position) algorithm
with atmospheric refraction corrections.
"""

from .golden_hour_engine import (
    GoldenHourEngine,
    SolarPosition,
    GoldenHourResult,
    get_golden_hour_engine,
)

__all__ = [
    "GoldenHourEngine",
    "SolarPosition",
    "GoldenHourResult",
    "get_golden_hour_engine",
]
