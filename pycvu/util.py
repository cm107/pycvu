from __future__ import annotations
import math
from typing import Literal, overload
import cv2
from PIL import Image as pilImage, \
    ImageDraw as pilImageDraw, \
    ImageFont as pilImageFont
import numpy as np

from .color import Color
from .vector import Vector

__all__ = [
    "Util",
    "CvUtil"
]

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
    def bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int ,int]:
        if type(color) is Color:
            return color
        return tuple(list(color)[::-1])

    @staticmethod
    def rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int ,int]:
        if type(color) is Color:
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
        else:
            raise TypeError(f"Unsupported type: {type(value).__name__}")

class CvUtil:
    @staticmethod
    def circle(
        img: np.ndarray,
        center: tuple[int, int] | Vector,
        radius: int,
        color: tuple[int, int, int] | Color,
        thickness: int, lineType: int
    ) -> np.ndarray:
        if type(center) is Vector:
            center = tuple(center)
        if type(color) is Color:
            color = color.bgr
        center = Util.cast_int(center)
        return cv2.circle(img, center, radius, color, thickness, lineType)
    
    @staticmethod
    def ellipse(
        img: np.ndarray,
        center: tuple[int, int] | Vector,
        axis: tuple[int, int] | Vector,
        angle: float, startAngle: float, endAngle: float,
        color: tuple[int, int, int] | Color,
        thickness: int, lineType: int
    ):
        if type(center) is Vector:
            center = tuple(center)
        center = Util.cast_int(center)
        if type(axis) is Vector:
            axis = tuple(axis)
        axis = Util.cast_int(axis)
        if type(color) is Color:
            color = color.bgr
        return cv2.ellipse(img, center, axis, angle, startAngle, endAngle, color, thickness, lineType)

    @staticmethod
    def rectangle(
        img: np.ndarray,
        pt1: tuple[int, int] | Vector,
        pt2: tuple[int, int] | Vector,
        color: tuple[int, int, int] | Color,
        thickness: int, lineType: int
    ) -> np.ndarray:
        if type(pt1) is Vector:
            pt1 = tuple(pt1)
        pt1 = Util.cast_int(pt1)
        if type(pt2) is Vector:
            pt2 = tuple(pt2)
        pt2 = Util.cast_int(pt2)
        if type(color) is Color:
            color = color.bgr
        return cv2.rectangle(img, pt1, pt2, color, thickness, lineType)

    @staticmethod
    def line(
        img: np.ndarray,
        pt1: tuple[int, int] | Vector,
        pt2: tuple[int, int] | Vector,
        color: tuple[int, int, int] | Color,
        thickness: int, lineType: int
    ) -> np.ndarray:
        if type(pt1) is Vector:
            pt1 = tuple(pt1)
        pt1 = Util.cast_int(pt1)
        if type(pt2) is Vector:
            pt2 = tuple(pt2)
        pt2 = Util.cast_int(pt2)
        if type(color) is Color:
            color = color.bgr
        return cv2.line(img, pt1, pt2, color, thickness, lineType)
    
    @staticmethod
    def affine_rotate(
        img: np.ndarray, angle: float, degrees: bool=True,
        scale: float=1, interpolation: int=cv2.INTER_LINEAR,
        adjustBorder: bool=False,
        center: tuple[int, int] | Vector=None,
        borderColor: tuple[int, int, int] | Color=[255, 255, 255],
    ) -> np.ndarray:
        if center is not None and type(center) is Vector:
            center = tuple(center)

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
        org: tuple[int, int] | Vector,
        fontFace: int=cv2.FONT_HERSHEY_SIMPLEX, fontScale: float=1,
        color: tuple[int, int, int] | Color = (255, 255, 255),
        thickness: int=None, lineType: int=None, bottomLeftOrigin: bool=False
    ) -> np.ndarray:
        if type(org) is Vector:
            org = tuple(org)
        org = Util.cast_int(org)
        if type(color) is Color:
            color = color.bgr
        return cv2.putText(
            img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin
        )

    @overload
    @staticmethod
    def resize(src: np.ndarray, dsize: tuple[int, int], interpolation: int=None) -> np.ndarray: ...

    @overload
    def resize(src: np.ndarray, fx: float, fy: float, interpolation: int=None) -> np.ndarray: ...

    @staticmethod
    def resize(
        src: np.ndarray,
        dsize: tuple[int, int] | Vector=None,
        fx: float=None, fy: float=None, interpolation: int=None
    ) -> np.ndarray:
        if dsize is not None:
            if type(dsize) is Vector:
                dsize = tuple(Vector)
            dsize = Util.cast_int(dsize)
        return cv2.resize(src, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)

class PilUtil:
    @staticmethod
    def text(
        img: pilImage.Image, text: str,
        fontPath: str, fontSize: int,
        color: tuple[int, int, int] | Color,
        position: tuple[float, float] | Vector,
        align: Literal['left', 'center', 'right']='left',
        direction: Literal['rtl', 'ltr', 'ttb']='ltr'
    ) -> pilImage.Image:
        if type(position) is Vector:
            position = tuple(position)
        if type(color) is Color:
            color = Util.cast_int(color.rgb)

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
        img: pilImage.Image, center: tuple[float, float], axis: tuple[float, float],
        fillColor: tuple[int, int, int] | Color=None,
        outlineColor: tuple[int, int, int] | Color=None,
        outlineWidth: float=None
    ) -> pilImage.Image:
        if type(fillColor) is Color:
            fillColor = Util.cast_int(fillColor.rgb)
        if type(outlineColor) is Color:
            outlineColor = Util.cast_int(outlineColor.rgb)

        p0 = (center[0] - axis[0], center[1] - axis[1])
        p1 = (center[0] + axis[0], center[1] + axis[1])
        shape: list[float] = list(p0) + list(p1)
        draw = pilImageDraw.Draw(img, mode=None)
        draw.ellipse(xy=shape, fill=fillColor, outline=outlineColor, width=outlineWidth)
        return img

    @staticmethod
    def circle(
        img: pilImage.Image, center: tuple[float, float], radius: float,
        fillColor: tuple[int, int, int] | Color=None,
        outlineColor: tuple[int, int, int] | Color=None,
        outlineWidth: float=None
    ) -> pilImage.Image:
        return PilUtil.ellipse(
            img=img, center=center, axis=(radius, radius),
            fillColor=fillColor, outlineColor=outlineColor,
            outlineWidth=outlineWidth
        )

    @staticmethod
    def hanko(
        img: pilImage.Image, text: str,
        fontPath: str, fontSize: int,
        color: tuple[int, int, int],
        position: tuple[float, float] | Vector,
        direction: Literal['rtl', 'ltr', 'ttb']='ttb',
        outlineWidth: float=20,
        marginOffset: float = 0, marginRatio: float = 0
    ):
        if type(position) is Vector:
            position = tuple(position)
        if type(color) is Color:
            color = Util.cast_int(color.rgb)

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