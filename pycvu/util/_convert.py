from __future__ import annotations
import copy
import os
from typing import Any, overload
import cv2
from PIL import Image as pilImage
import numpy as np
import numpy.typing as npt
from pyevu.vector2 import Vector2

from pycvu.interval import Interval

from ..base import ContextVarRef
from ..color import Color, HSV
from ..vector import Vector
from ..text_generator import TextGenerator, TextSampler, TextComposer
from ._var import *
from ._loadable_image import ImageVar, \
    LoadableImage, LoadableImageHandler, \
    LoadableImageMask, LoadableImageMaskHandler

__all__ = ["Convert"]

class Convert:
    @staticmethod
    def cv_to_pil(img: np.ndarray) -> pilImage.Image:
        result = img.copy()
        if result.ndim == 2: # grayscale
            pass
        else:
            assert result.ndim == 3
            if result.shape[2] == 3: # rgb
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            elif result.shape[2] == 4: # rgba
                result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
            else:
                raise ValueError
        return pilImage.fromarray(result)
    
    @staticmethod
    def pil_to_cv(img: pilImage.Image) -> np.ndarray:
        result = np.array(img, dtype=np.uint8)
        if result.ndim == 2: # grayscale
            pass
        else:
            assert result.ndim == 3
            if result.shape[2] == 3: # rgb
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            elif result.shape[2] == 4: # rgba
                result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
            else:
                raise ValueError
        return result
    
    @staticmethod
    def bgr_to_rgb(color: ColorVar) -> tuple[int, int ,int] | Color | Interval:
        if type(color) in [Color, Interval]:
            return color
        return tuple(list(color)[::-1])

    @staticmethod
    def rgb_to_bgr(color: ColorVar) -> tuple[int, int ,int] | Color | Interval:
        if type(color) in [Color, Interval]:
            return color
        return tuple(list(color)[::-1])

    @overload
    @staticmethod
    def cast_int(value: tuple[float, float]) -> tuple[int, int]: ...

    @overload
    @staticmethod
    def cast_int(value: Vector) -> Vector: ...

    @staticmethod
    def cast_int(value: tuple | Vector | Color) -> tuple | Vector | Color:
        if type(value) is tuple:
            return tuple([int(val) for val in value])
        elif type(value) is Vector:
            return Vector(int(value.x), int(value.y))
        elif type(value) is Color:
            result = value.copy()
            for attr in ['r', 'g', 'b']:
                setattr(result, attr, int(getattr(result, attr)))
            return result
        elif value is None:
            return None
        elif type(value) in [int, float]:
            return int(value)
        else:
            raise TypeError(f"Unsupported type: {type(value).__name__}")

    @staticmethod
    def cast_vector(value: VectorVar) -> tuple[int, int]:
        result = copy.deepcopy(value)
        if type(result) in [Vector, Vector2]:
            result = tuple(result)
        elif type(result) is Interval:
            result = tuple(result.random())
        elif value is None:
            return None
        result = Convert.cast_int(result)
        return result

    @staticmethod
    def cast_color(
        value: ColorVar,
        tupleAttr: str='bgr', asInt: bool=False
    ) -> tuple[int, int, int]:
        result = copy.deepcopy(value)
        if type(result) is Color:
            result = getattr(result, tupleAttr)
        elif type(result) is HSV:
            result = getattr(result.to_color(), tupleAttr)
        elif type(result) is Interval:
            return Convert.cast_color(
                value=result.random(),
                tupleAttr=tupleAttr, asInt=asInt
            )
        elif type(result) is tuple:
            pass
        elif value is None:
            return None
        else:
            raise TypeError(f"Unsupported type: {type(value).__name__}")
        if asInt:
            result = Convert.cast_int(result)
        return result

    @staticmethod
    def cast_str(value: StringVar) -> str:
        if type(value) is str:
            return value
        elif type(value) in [TextGenerator, TextSampler, TextComposer]:
            return value.random()
        else:
            raise TypeError

    @staticmethod
    def cast_builtin(value: Any | Interval, asInt: bool=False) -> Any:
        result = copy.deepcopy(value)
        if type(value) is Interval:
            result = result.random()
        if asInt:
            result = Convert.cast_int(result)
        return result
    
    @staticmethod
    def cast_image(value: ImageVar) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        """Gets the image and mask from the corresponding type."""
        if type(value) is ContextVarRef:
            value = value.value
        if type(value) is np.ndarray:
            return value, None
        elif type(value) is LoadableImage:
            return value._img, None
        elif type(value) is LoadableImageHandler:
            obj = value.random()
            return obj._img, None
        elif type(value) is LoadableImageMask:
            return value._img, value._mask
        elif type(value) is LoadableImageMaskHandler:
            obj = value.random()
            return obj._img, obj._mask
        else:
            raise TypeError

    @staticmethod
    def cast_image_input(value: ImageInput) -> npt.NDArray[np.uint8]:
        if type(value) is np.ndarray:
            return value
        elif type(value) is str:
            if not os.path.isfile(value):
                raise FileNotFoundError(f"Failed to find image at: {value}")
            return cv2.imread(value)
        else:
            raise TypeError
