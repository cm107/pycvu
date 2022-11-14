from typing import TypeVar
NumberVar = float | int

NV = TypeVar('NV', bound=NumberVar)

__all__ = [
    'clamp'
]

def clamp(value: NV, minValue: NV, maxValue: NV) -> NV:
    return min(max(value, minValue), maxValue)
