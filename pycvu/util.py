from __future__ import annotations
import math
import copy
from typing import Any, Callable, Literal, overload
import cv2
from PIL import Image as pilImage, \
    ImageDraw as pilImageDraw, \
    ImageFont as pilImageFont
import numpy as np

from pycvu.interval import Interval

from .color import Color, HSV
from .vector import Vector

__all__ = [
    "Util",
    "CvUtil",
    "PilUtil"
]

VectorVar = tuple[int, int] | Vector | Interval
ImageVectorCallback = Callable[[np.ndarray], VectorVar]
PilImageVectorCallback = Callable[[pilImage.Image], VectorVar]
ColorVar = tuple[int, int, int] | Color | HSV | Interval
IntVar = int | Interval
FloatVar = float | Interval

class Util:
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
        if type(result) is Vector:
            result = tuple(result)
        elif type(result) is Interval:
            result = tuple(result.random())
        elif value is None:
            return None
        result = Util.cast_int(result)
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
            return Util.cast_color(
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
            result = Util.cast_int(result)
        return result

    @staticmethod
    def cast_builtin(value: Any | Interval, asInt: bool=False) -> Any:
        result = copy.deepcopy(value)
        if type(value) is Interval:
            result = result.random()
        if asInt:
            result = Util.cast_int(result)
        return result

class CvUtil:
    @staticmethod
    def circle(
        img: np.ndarray,
        center: VectorVar | ImageVectorCallback,
        radius: IntVar,
        color: ColorVar,
        thickness: IntVar,
        lineType: int
    ) -> np.ndarray:
        if callable(center):
            center = center(img)
        center = Util.cast_vector(center)
        radius = Util.cast_builtin(radius)
        color = Util.cast_color(color)
        thickness = Util.cast_builtin(thickness)
        return cv2.circle(img, center, radius, color, thickness, lineType)
    
    @staticmethod
    def ellipse(
        img: np.ndarray,
        center: VectorVar | ImageVectorCallback,
        axis: VectorVar,
        angle: FloatVar,
        startAngle: FloatVar,
        endAngle: FloatVar,
        color: ColorVar,
        thickness: IntVar,
        lineType: int
    ):
        if callable(center):
            center = center(img)
        center = Util.cast_vector(center)
        axis = Util.cast_vector(axis)
        angle = Util.cast_builtin(angle)
        startAngle = Util.cast_builtin(startAngle)
        endAngle = Util.cast_builtin(endAngle)
        if startAngle > endAngle:
            tmp = startAngle; startAngle = endAngle
            endAngle = tmp
        color = Util.cast_color(color)
        thickness = Util.cast_builtin(thickness)
        return cv2.ellipse(img, center, axis, angle, startAngle, endAngle, color, thickness, lineType)

    @staticmethod
    def rectangle(
        img: np.ndarray,
        pt1: VectorVar | ImageVectorCallback,
        pt2: VectorVar | ImageVectorCallback,
        color: ColorVar,
        thickness: IntVar,
        lineType: int
    ) -> np.ndarray:
        if callable(pt1):
            pt1 = pt1(img)
        if callable(pt2):
            pt2 = pt2(img)
        pt1 = Util.cast_vector(pt1)
        pt2 = Util.cast_vector(pt2)
        color = Util.cast_color(color)
        thickness = Util.cast_builtin(thickness)
        return cv2.rectangle(img, pt1, pt2, color, thickness, lineType)

    @staticmethod
    def line(
        img: np.ndarray,
        pt1: VectorVar | ImageVectorCallback,
        pt2: VectorVar | ImageVectorCallback,
        color: ColorVar,
        thickness: IntVar,
        lineType: int
    ) -> np.ndarray:
        if callable(pt1):
            pt1 = pt1(img)
        if callable(pt2):
            pt2 = pt2(img)
        pt1 = Util.cast_vector(pt1)
        pt2 = Util.cast_vector(pt2)
        color = Util.cast_color(color)
        thickness = Util.cast_builtin(thickness)
        return cv2.line(img, pt1, pt2, color, thickness, lineType)
    
    @staticmethod
    def affine_rotate(
        img: np.ndarray,
        angle: FloatVar,
        degrees: bool=True,
        scale: FloatVar=1,
        interpolation: int=cv2.INTER_LINEAR,
        adjustBorder: bool=False,
        center: VectorVar=None,
        borderColor: ColorVar=(255, 255, 255),
    ) -> np.ndarray:
        angle = Util.cast_builtin(angle)
        scale = Util.cast_builtin(scale)
        center = Util.cast_vector(center)
        borderColor = Util.cast_color(borderColor)

        h, w = img.shape[:2]
        if degrees:
            angle = math.radians(angle)
        if not adjustBorder:
            sizeRotation: tuple[int, int] = (w, h)
        else:
            wRot = int(h * abs(math.sin(angle)) + w * abs(math.cos(angle)))
            hRot = int(h * abs(math.cos(angle)) + w * abs(math.sin(angle)))
            sizeRotation: tuple[int, int] = (wRot, hRot)
        if center is None:
            center = (int(w/2), int(h/2))
        affineMatrix = cv2.getRotationMatrix2D(center, math.degrees(angle), scale)
        if adjustBorder:
            affineMatrix[0][2] = affineMatrix[0][2] - w/2 + wRot/2
            affineMatrix[1][2] = affineMatrix[1][2] - h/2 + hRot/2
        if type(borderColor) is Color:
            borderColor = borderColor.bgr
        return cv2.warpAffine(img, affineMatrix, sizeRotation, flags=interpolation, borderValue=borderColor)
    
    def text(
        img: np.ndarray, text: str,
        org: VectorVar,
        fontFace: int=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale: FloatVar=1,
        color: ColorVar = (255, 255, 255),
        thickness: IntVar=None,
        lineType: int=None, bottomLeftOrigin: bool=False
    ) -> np.ndarray:
        org = Util.cast_vector(org)
        fontScale = Util.cast_builtin(fontScale)
        color = Util.cast_color(color)
        thickness = Util.cast_builtin(thickness)
        return cv2.putText(
            img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin
        )

    @overload
    @staticmethod
    def resize(
        src: np.ndarray,
        dsize: VectorVar,
        interpolation: int=None
    ) -> np.ndarray: ...

    @overload
    def resize(
        src: np.ndarray,
        fx: FloatVar, fy: FloatVar,
        interpolation: int=None
    ) -> np.ndarray: ...

    @staticmethod
    def resize(
        src: np.ndarray,
        dsize: VectorVar=None,
        fx: FloatVar=None, fy: FloatVar=None,
        interpolation: int=None
    ) -> np.ndarray:
        dsize = Util.cast_vector(dsize)
        fx = Util.cast_builtin(fx)
        fy = Util.cast_builtin(fy)
        return cv2.resize(src, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)

    class Callback:
        @staticmethod
        def get_position_interval(img: np.ndarray) -> Interval[Vector[float]]:
            h, w = img.shape[:2]
            return Interval[Vector[float]](
                Vector[float](0, 0),
                Vector[float](w - 1, h - 1)
            )

class PilUtil:
    @staticmethod
    def text(
        img: pilImage.Image, text: str,
        fontPath: str,
        fontSize: IntVar,
        color: ColorVar,
        position: VectorVar | PilImageVectorCallback,
        align: Literal['left', 'center', 'right']='left',
        direction: Literal['rtl', 'ltr', 'ttb']='ltr'
    ) -> pilImage.Image:
        fontSize = Util.cast_builtin(fontSize, asInt=True)
        color = Util.cast_color(color, tupleAttr='rgb', asInt=True)
        if callable(position):
            position = position(img)
        position = Util.cast_vector(position)

        draw = pilImageDraw.Draw(img, mode=None)
        font = pilImageFont.truetype(font=fontPath, size=fontSize)
        w, h = draw.textsize(text, font=font, direction=direction)
        position = (position[0] - w/2, position[1] - h/2)

        draw.text(
            xy=position,
            text=text,
            fill=color,
            font=font,
            align=align,
            direction=direction
        )
        return img

    @staticmethod
    def ellipse(
        img: pilImage.Image,
        center: VectorVar | PilImageVectorCallback,
        axis: VectorVar,
        fillColor: ColorVar=None,
        outlineColor: ColorVar=None,
        outlineWidth: IntVar=None
    ) -> pilImage.Image:
        if callable(center):
            center = center(img)
        center = Util.cast_vector(center)
        axis = Util.cast_vector(axis)
        fillColor = Util.cast_color(fillColor, asInt=True)
        outlineColor = Util.cast_color(outlineColor, asInt=True)
        outlineWidth = Util.cast_builtin(outlineWidth)

        p0 = (center[0] - axis[0], center[1] - axis[1])
        p1 = (center[0] + axis[0], center[1] + axis[1])
        shape: list[float] = list(p0) + list(p1)
        draw = pilImageDraw.Draw(img, mode=None)
        draw.ellipse(xy=shape, fill=fillColor, outline=outlineColor, width=outlineWidth)
        return img

    @staticmethod
    def circle(
        img: pilImage.Image,
        center: VectorVar | PilImageVectorCallback,
        radius: FloatVar,
        fillColor: ColorVar=None,
        outlineColor: ColorVar=None,
        outlineWidth: IntVar=None
    ) -> pilImage.Image:
        radius = Util.cast_builtin(radius)
        return PilUtil.ellipse(
            img=img, center=center, axis=(radius, radius),
            fillColor=fillColor, outlineColor=outlineColor,
            outlineWidth=outlineWidth
        )

    @staticmethod
    def hanko(
        img: pilImage.Image, text: str,
        fontPath: str,
        fontSize: IntVar,
        color: ColorVar,
        position: VectorVar | PilImageVectorCallback,
        direction: Literal['rtl', 'ltr', 'ttb']='ttb',
        outlineWidth: IntVar=20,
        marginOffset: FloatVar = 0,
        marginRatio: FloatVar = 0
    ):
        fontSize = Util.cast_builtin(fontSize)
        color = Util.cast_color(color, tupleAttr='rgb', asInt=True)
        if callable(position):
            position = position(img)
        position = Util.cast_vector(position)
        outlineWidth = Util.cast_builtin(outlineWidth)
        marginOffset = Util.cast_builtin(marginOffset)
        marginRatio = Util.cast_builtin(marginRatio)

        draw = pilImageDraw.Draw(img, mode=None)
        font = pilImageFont.truetype(font=fontPath, size=fontSize)
        w, h = draw.textsize(text, font=font, direction=direction)
        r = int((w**2 + h**2)**0.5) * 0.5
        r += marginRatio * r
        r += marginOffset

        draw = pilImageDraw.Draw(img, mode=None)
        font = pilImageFont.truetype(font=fontPath, size=fontSize)
        w, h = draw.textsize(text, font=font, direction=direction)
        textPosition = (position[0] - w/2, position[1] - h/2)

        draw.text(
            xy=textPosition, text=text, fill=color,
            font=font, direction=direction
        )
        PilUtil.circle(
            img, center=position, radius=r,
            fillColor=None, outlineColor=color,
            outlineWidth=outlineWidth
        )
        return img

    class Callback:
        @staticmethod
        def get_position_interval(img: pilImage.Image) -> Interval[Vector[float]]:
            w = img.width; h = img.height
            return Interval[Vector[float]](
                Vector[float](0, 0),
                Vector[float](w - 1, h - 1)
            )

    @staticmethod
    def debug():
        img = pilImage.new("RGB", (500, 500))
        
        import pycvu
        fontPath = f"{pycvu.__path__[0]}/data/font/ipaexg.ttf"
        PilUtil.hanko(
            img, text="夏生", fontPath=fontPath, fontSize=150, color=(255, 0, 0),
            position=(250, 250), direction='ttb', outlineWidth=20,
            marginOffset=0, marginRatio=0.1
        )
        img.show()