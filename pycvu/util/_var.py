from __future__ import annotations
from typing import Callable
from functools import partial
import cv2
from PIL import Image as pilImage
import numpy as np
import numpy.typing as npt

from pycvu.interval import Interval

from ..color import Color, HSV
from ..vector import Vector
from ..text_generator import TextGenerator, TextSampler
from ..base import BaseUtil

__all__ = [
    "StringVar",
    "VectorVar",
    "ImageVectorCallback",
    "PilImageVectorCallback",
    "ColorVar",
    "IntVar",
    "FloatVar",
    "RepeatDrawCallback",
    "DrawCallback"
]

StringVar = str | TextGenerator | TextSampler
VectorVar = tuple[int, int] | Vector | Interval
ImageVectorCallback = Callable[[np.ndarray], VectorVar]
PilImageVectorCallback = Callable[[pilImage.Image], VectorVar]
ColorVar = tuple[int, int, int] | Color | HSV | Interval
IntVar = int | Interval
FloatVar = float | Interval

class RepeatDrawCallback:
    def __init__(self, p: partial, repeat: int=1):
        self.p = p
        self.repeat = repeat
    
    def to_dict(self) -> dict:
        return dict(
            p=BaseUtil.to_func_dict(self.p),
            repeat=self.repeat,
            _typedict=BaseUtil.to_type_dict(type(self))
        )
    
    @classmethod
    def from_dict(cls, item_dict: dict, **kwargs) -> RepeatDrawCallback:
        return RepeatDrawCallback(
            p=BaseUtil.from_func_dict(item_dict['p'], **kwargs),
            repeat=item_dict['repeat']
        )

DrawCallback = Callable[[np.ndarray], np.ndarray] | partial | RepeatDrawCallback
