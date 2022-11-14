from __future__ import annotations
from typing import Callable
from functools import partial
import random
from PIL import Image as pilImage
import numpy as np
import numpy.typing as npt

from pycvu.interval import Interval

from ..color import Color, HSV
from ..vector import Vector
from ..text_generator import TextGenerator, TextSampler
from ..base import BaseUtil
from ._func import clamp

__all__ = [
    "StringVar",
    "VectorVar",
    "ImageVectorCallback",
    "PilImageVectorCallback",
    "ColorVar",
    "NoiseVar",
    "IntVar",
    "FloatVar",
    "ImageInput",
    "DrawCallback"
]

StringVar = str | TextGenerator | TextSampler
VectorVar = tuple[int, int] | Vector | Interval
ImageVectorCallback = Callable[[np.ndarray], VectorVar]
PilImageVectorCallback = Callable[[pilImage.Image], VectorVar]
ColorVar = tuple[int, int, int] | Color | HSV | Interval
NoiseVar = Interval[Color] | Interval[HSV] | Interval[int]
IntVar = int | Interval
FloatVar = float | Interval
ImageInput = npt.NDArray[np.uint8] | str
DrawCallback = Callable[[np.ndarray], np.ndarray] | partial
