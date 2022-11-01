from __future__ import annotations
from typing import TypeVar
from .base import Base

def clamp(value: float, minVal: float, maxVal: float) -> float:
    return max(min(value, minVal), maxVal)

class Color(Base):
    def __init__(self, r: float=0, g: float=0, b: float=0, a: float=255, _scale: float=255):
        self.r = r; self.g = g; self.b = b; self.a = a
        self._scale = _scale

    def __str__(self) -> str:
        innerStr = ', '.join([
            f'{key}={int(val)}'
            for key, val in self.__dict__.items()
            if (
                not key.startswith('_')
                and not (
                    key == 'a'
                    and self.a == self._scale
                )
            )
        ])
        return f"{type(self).__name__}({innerStr})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def scale(self) -> float:
        return self._scale
    
    @scale.setter
    def scale(self, value: float):
        ratio = value / self._scale
        self.r *= ratio
        self.g *= ratio
        self.b *= ratio
        self.a *= ratio
        self._scale = value

    def rescale(self, newScale: float) -> Color:
        self.scale = newScale
        return self

    @property
    def rgb(self) -> tuple[float, float, float]:
        return (self.r, self.g, self.b)
    
    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    @property
    def bgr(self) -> tuple[float, float, float]:
        return (self.b, self.g, self.r)
    
    @property
    def bgra(self) -> tuple[float, float, float, float]:
        return (self.b, self.g, self.r, self.a)
    
    @classmethod
    @property
    def red(cls) -> Color:
        return cls(255, 0, 0)
    
    @classmethod
    @property
    def green(cls) -> Color:
        return cls(0, 255, 0)
    
    @classmethod
    @property
    def blue(cls) -> Color:
        return cls(0, 255, 0)

    @classmethod
    @property
    def black(cls) -> Color:
        return cls(0, 0, 0)
    
    @classmethod
    @property
    def white(cls) -> Color:
        return cls(255, 255, 255)
