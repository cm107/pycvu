from typing import TypeVar, Callable, Any
import inspect

NumberVar = float | int

NV = TypeVar('NV', bound=NumberVar)

__all__ = [
    'clamp'
]

def clamp(value: NV, minValue: NV, maxValue: NV) -> NV:
    return min(max(value, minValue), maxValue)

T = TypeVar('T')

def argmax(
    vals: list[T],
    func: Callable[[T], Any] | Callable[[int, T], Any]=None
) -> int | None:
    maxIdx: int = None
    maxValue: Any = None
    if func is None:
        for i, val in enumerate(vals):
            if maxValue is None or val > maxValue:
                maxIdx = i; maxValue = val
    else:
        if len(inspect.getargspec(func).args) == 1:
            for i, _val in enumerate(vals):
                val = func(_val)
                if maxValue is None or val > maxValue:
                    maxIdx = i; maxValue = val
        elif len(inspect.getargspec(func).args) == 2:
            for i, _val in enumerate(vals):
                val = func(i, _val)
                if maxValue is None or val > maxValue:
                    maxIdx = i; maxValue = val
        else:
            raise ValueError('Invalid number of args in func.')
    return maxIdx
