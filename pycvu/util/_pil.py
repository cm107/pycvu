from __future__ import annotations
from typing import Literal, TYPE_CHECKING, Callable
from functools import partial
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
    def _apply_rotate(
        img: pilImage.Image,
        rotation: float,
        drawCallback: Callable[[pilImageDraw.ImageDraw],]=None,
        center: tuple[float, float]=None
    ) -> pilImage.Image:
        tmpImg = pilImage.new(
            'RGBA', (img.width, img.height), color=(0, 0, 0, 0)
        )
        if drawCallback is not None:
            tmpImgDraw = pilImageDraw.Draw(tmpImg)
            drawCallback(tmpImgDraw)

        tmpImg = tmpImg.rotate(angle=rotation, center=center, expand=False)
        img.paste(tmpImg, box=(0, 0, img.width, img.height), mask=tmpImg)
        return img

    @staticmethod
    def text(
        img: pilImage.Image, text: StringVar,
        fontPath: str,
        fontSize: IntVar,
        color: ColorVar,
        position: VectorVar | PilImageVectorCallback,
        align: Literal['left', 'center', 'right']='left',
        direction: Literal['rtl', 'ltr', 'ttb']='ltr',
        rotation: FloatVar=0,
        refMask: Mask=None
    ) -> pilImage.Image:
        text = Convert.cast_str(text)
        fontSize = Convert.cast_builtin(fontSize, asInt=True)
        color = Convert.cast_color(color, tupleAttr='rgb', asInt=True)
        if callable(position):
            position = position(img)
        position = Convert.cast_vector(position)
        rotation = Convert.cast_builtin(rotation)

        draw = pilImageDraw.Draw(img, mode=None)
        font = pilImageFont.truetype(font=fontPath, size=fontSize)
        w, h = draw.textsize(text, font=font, direction=direction)
        xy = (position[0] - w/2, position[1] - h/2)

        def drawCallback(d: pilImageDraw.ImageDraw, c: tuple[int, int, int]):
            d.text(
                xy=xy,
                text=text,
                fill=c,
                font=font,
                align=align,
                direction=direction
            )
        
        if refMask is not None:
            mask = pilImage.new("RGB", (img.width, img.height), color=(0, 0, 0))
            maskColor = (255, 255, 255)
            if rotation != 0:
                mask = PilUtil._apply_rotate(mask, rotation, partial(drawCallback, c=maskColor), center=position)
            else:
                drawCallback(pilImageDraw.Draw(mask, mode=None), maskColor)
            mask = Convert.pil_to_cv(mask)
            mask = MaskUtil.eq_color(mask, color=maskColor)
            refMask._mask = mask

        if rotation != 0:
            img = PilUtil._apply_rotate(img, rotation, partial(drawCallback, c=color), center=position)
        else:
            drawCallback(draw, color)
        return img

    @staticmethod
    def ellipse(
        img: pilImage.Image,
        center: VectorVar | PilImageVectorCallback,
        axis: VectorVar,
        fillColor: ColorVar=None,
        outlineColor: ColorVar=None,
        outlineWidth: IntVar=None,
        rotation: FloatVar=0,
        refMask: Mask=None
    ) -> pilImage.Image:
        if callable(center):
            center = center(img)
        center = Convert.cast_vector(center)
        axis = Convert.cast_vector(axis)
        fillColor = Convert.cast_color(fillColor, asInt=True)
        outlineColor = Convert.cast_color(outlineColor, asInt=True)
        outlineWidth = Convert.cast_builtin(outlineWidth)
        rotation = Convert.cast_builtin(rotation)

        p0 = (center[0] - axis[0], center[1] - axis[1])
        p1 = (center[0] + axis[0], center[1] + axis[1])
        shape: list[float] = list(p0) + list(p1)

        def drawCallback(d: pilImageDraw.ImageDraw, fill: tuple=None, outline: tuple=None):
            d.ellipse(
                xy=shape,
                fill=fill if fill is not None else None,
                outline=outline if outline is not None else None,
                width=outlineWidth
            )

        if refMask is not None:
            mask = pilImage.new("RGB", (img.width, img.height), color=(0, 0, 0))
            maskColor = (255, 255, 255)
            if rotation != 0:
                mask = PilUtil._apply_rotate(
                    mask, rotation, partial(
                        fill=maskColor if fillColor is not None else None,
                        outline=maskColor if outlineColor is not None else None
                    ),
                    center=center
                )
            else:
                drawCallback(
                    pilImageDraw.Draw(mask, mode=None),
                    fill=maskColor if fillColor is not None else None,
                    outline=maskColor if outlineColor is not None else None
                )
            mask = Convert.pil_to_cv(mask)
            mask = MaskUtil.eq_color(mask, color=maskColor)
            refMask._mask = mask

        draw = pilImageDraw.Draw(img, mode=None)
        if rotation != 0:
            img = PilUtil._apply_rotate(
                img, rotation,
                partial(drawCallback, fill=fillColor, outline=outlineColor),
                center=center
            )
        else:
            drawCallback(draw, fill=fillColor, outline=outlineColor)
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
        outlineWidthRatio: FloatVar=20,
        marginOffset: FloatVar = 0,
        marginRatio: FloatVar = 0,
        rotation: FloatVar=0,
        refMask: Mask=None
    ):
        text = Convert.cast_str(text)
        fontSize = Convert.cast_builtin(fontSize)
        color = Convert.cast_color(color, tupleAttr='rgb', asInt=True)
        if callable(position):
            position = position(img)
        position = Convert.cast_vector(position)
        outlineWidthRatio = Convert.cast_builtin(outlineWidthRatio)
        marginOffset = Convert.cast_builtin(marginOffset)
        marginRatio = Convert.cast_builtin(marginRatio)
        rotation = Convert.cast_builtin(rotation)

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

        def textDrawCallback(d: pilImageDraw.ImageDraw, fill: tuple):
            d.text(
                xy=textPosition, text=text, fill=fill,
                font=font, direction=direction
            )

        if rotation != 0:
            img = PilUtil._apply_rotate(
                img, rotation,
                partial(textDrawCallback, fill=color),
                center=position
            )
        else:
            textDrawCallback(draw, fill=color)
        img = PilUtil.circle(
            img, center=position, radius=r,
            fillColor=None, outlineColor=color,
            outlineWidth=int(outlineWidthRatio * r),
            refMask=refMask
        )
        if refMask is not None:
            textMask = pilImage.new("RGB", (img.width, img.height), color=(0, 0, 0))
            textMaskColor = (255, 255, 255)
            if rotation != 0:
                textMask = PilUtil._apply_rotate(
                    textMask, rotation,
                    partial(textDrawCallback, fill=textMaskColor),
                    center=position
                )
            else:
                textDrawCallback(pilImageDraw.Draw(textMask, mode=None), fill=textMaskColor)
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
            position=(250, 250), direction='ttb', outlineWidthRatio=20,
            marginOffset=0, marginRatio=0.1
        )
        img.show()
    
    @staticmethod
    def debug_text():
        img = pilImage.new("RGB", (500, 500), color=(0, 0, 255))
        
        import pycvu
        fontPath = f"{pycvu.__path__[0]}/data/font/ipaexg.ttf"
        img = PilUtil.text(
            img=img, text='hello', fontPath=fontPath, fontSize=150,
            color=(255, 0, 0), position=(250, 250), rotation=45
        )
        img.show()

    @staticmethod
    def debug_ellipse():
        img = pilImage.new("RGB", (500, 500), color=(0, 0, 255))
        img = PilUtil.ellipse(
            img=img, center=(300, 350), axis=(100, 50),
            fillColor=(0,255,0), outlineColor=(255,0,0),
            outlineWidth=10, rotation=45
        )
        img.show()