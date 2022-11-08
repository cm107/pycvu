from __future__ import annotations
from typing import Literal, TYPE_CHECKING
from PIL import Image as pilImage, \
    ImageDraw as pilImageDraw, \
    ImageFont as pilImageFont

from pycvu.interval import Interval

from ..vector import Vector
if TYPE_CHECKING:
    from ..mask import Mask

__all__ = [
    "PilUtil"
]

from ._var import *
from ._convert import Convert
from ._mask import MaskUtil

class PilUtil:
    @staticmethod
    def text(
        img: pilImage.Image, text: StringVar,
        fontPath: str,
        fontSize: IntVar,
        color: ColorVar,
        position: VectorVar | PilImageVectorCallback,
        align: Literal['left', 'center', 'right']='left',
        direction: Literal['rtl', 'ltr', 'ttb']='ltr',
        refMask: Mask=None
    ) -> pilImage.Image:
        text = Convert.cast_str(text)
        fontSize = Convert.cast_builtin(fontSize, asInt=True)
        color = Convert.cast_color(color, tupleAttr='rgb', asInt=True)
        if callable(position):
            position = position(img)
        position = Convert.cast_vector(position)

        draw = pilImageDraw.Draw(img, mode=None)
        font = pilImageFont.truetype(font=fontPath, size=fontSize)
        w, h = draw.textsize(text, font=font, direction=direction)
        position = (position[0] - w/2, position[1] - h/2)

        if refMask is not None:
            mask = pilImage.new("RGB", (img.width, img.height), color=(0, 0, 0))
            drawMask = pilImageDraw.Draw(mask, mode=None)
            maskColor = (255, 255, 255)
            drawMask.text(
                xy=position,
                text=text,
                fill=maskColor,
                font=font,
                align=align,
                direction=direction
            )
            mask = Convert.pil_to_cv(mask)
            mask = MaskUtil.eq_color(mask, color=maskColor)
            refMask._mask = mask

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
        outlineWidth: IntVar=None,
        refMask: Mask=None
    ) -> pilImage.Image:
        if callable(center):
            center = center(img)
        center = Convert.cast_vector(center)
        axis = Convert.cast_vector(axis)
        fillColor = Convert.cast_color(fillColor, asInt=True)
        outlineColor = Convert.cast_color(outlineColor, asInt=True)
        outlineWidth = Convert.cast_builtin(outlineWidth)

        p0 = (center[0] - axis[0], center[1] - axis[1])
        p1 = (center[0] + axis[0], center[1] + axis[1])
        shape: list[float] = list(p0) + list(p1)
        draw = pilImageDraw.Draw(img, mode=None)

        if refMask is not None:
            mask = pilImage.new("RGB", (img.width, img.height), color=(0, 0, 0))
            drawMask = pilImageDraw.Draw(mask, mode=None)
            maskColor = (255, 255, 255)
            drawMask.ellipse(
                xy=shape,
                fill=maskColor if fillColor is not None else None,
                outline=maskColor if outlineColor is not None else None,
                width=outlineWidth
            )
            mask = Convert.pil_to_cv(mask)
            mask = MaskUtil.eq_color(mask, color=maskColor)
            refMask._mask = mask

        draw.ellipse(xy=shape, fill=fillColor, outline=outlineColor, width=outlineWidth)
        return img

    @staticmethod
    def circle(
        img: pilImage.Image,
        center: VectorVar | PilImageVectorCallback,
        radius: FloatVar,
        fillColor: ColorVar=None,
        outlineColor: ColorVar=None,
        outlineWidth: IntVar=None,
        refMask: Mask=None
    ) -> pilImage.Image:
        radius = Convert.cast_builtin(radius)
        return PilUtil.ellipse(
            img=img, center=center, axis=(radius, radius),
            fillColor=fillColor, outlineColor=outlineColor,
            outlineWidth=outlineWidth,
            refMask=refMask
        )

    @staticmethod
    def hanko(
        img: pilImage.Image, text: StringVar,
        fontPath: str,
        fontSize: IntVar,
        color: ColorVar,
        position: VectorVar | PilImageVectorCallback,
        direction: Literal['rtl', 'ltr', 'ttb']='ttb',
        outlineWidth: IntVar=20,
        marginOffset: FloatVar = 0,
        marginRatio: FloatVar = 0,
        refMask: Mask=None
    ):
        text = Convert.cast_str(text)
        fontSize = Convert.cast_builtin(fontSize)
        color = Convert.cast_color(color, tupleAttr='rgb', asInt=True)
        if callable(position):
            position = position(img)
        position = Convert.cast_vector(position)
        outlineWidth = Convert.cast_builtin(outlineWidth)
        marginOffset = Convert.cast_builtin(marginOffset)
        marginRatio = Convert.cast_builtin(marginRatio)

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
            outlineWidth=outlineWidth,
            refMask=refMask
        )
        if refMask is not None:
            textMask = pilImage.new("RGB", (img.width, img.height), color=(0, 0, 0))
            drawTextMask = pilImageDraw.Draw(textMask, mode=None)
            textMaskColor = (255, 255, 255)
            drawTextMask.text(
                xy=textPosition, text=text, fill=textMaskColor,
                font=font, direction=direction
            )
            textMask = Convert.pil_to_cv(textMask)
            textMask = MaskUtil.eq_color(textMask, color=textMaskColor)
            assert refMask._mask is not None
            refMask._mask |= textMask
            
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
