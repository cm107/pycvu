from __future__ import annotations
from typing import Literal, TYPE_CHECKING, Callable
from functools import partial
from PIL import Image as pilImage, \
    ImageDraw as pilImageDraw, \
    ImageFont as pilImageFont

from pycvu.interval import Interval
import pycvu
from pyevu import BBox2D, Vector2

from ..vector import Vector
if TYPE_CHECKING:
    from ..mask import Mask, MaskHandler

__all__ = [
    "PilUtil"
]

from ._var import *
from ._convert import Convert
from ._mask import MaskUtil

class PilUtil:
    defaultFontPath = f"{pycvu.__path__[0]}/data/font/ipaexg.ttf"

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
        ensureNoOverlap: bool=False,
        ensureInBounds: bool=False,
        fillMaskTextbox: bool=False,
        refMask: Mask=None,
        maskHandler: MaskHandler=None
    ) -> pilImage.Image:
        while True:
            _text = Convert.cast_str(text)
            _fontSize = Convert.cast_builtin(fontSize, asInt=True)
            _color = Convert.cast_color(color, tupleAttr='rgb', asInt=True)
            if callable(position):
                _position = position(img)
            else:
                _position = position
            _position = Convert.cast_vector(_position)
            _rotation = Convert.cast_builtin(rotation)

            draw = pilImageDraw.Draw(img, mode=None)
            font = pilImageFont.truetype(font=fontPath, size=_fontSize)
            w, h = draw.textsize(_text, font=font, direction=direction)
            xy = (_position[0] - w/2, _position[1] - h/2)

            if ensureInBounds:
                if _position[0] - w/2 < 0:
                    continue
                if _position[1] - h/2 < 0:
                    continue
                if _position[0] + w/2 > img.width:
                    continue
                if _position[1] + h/2 > img.height:
                    continue

            def drawCallback(d: pilImageDraw.ImageDraw, c: tuple[int, int, int], fillTextbox: bool=False):
                if not fillTextbox:
                    d.text(
                        xy=xy,
                        text=_text,
                        fill=c,
                        font=font,
                        align=align,
                        direction=direction,
                    )
                else:
                    p0 = (_position[0] - w/2, _position[1] - h/2)
                    p1 = (_position[0] + w/2, _position[1] + h/2)
                    shape: list[float] = list(p0) + list(p1)
                    d.rectangle(xy=shape, fill=c)
            
            if refMask is not None:
                mask = pilImage.new("RGB", (img.width, img.height), color=(0, 0, 0))
                maskColor = (255, 255, 255)
                if _rotation != 0:
                    mask = PilUtil._apply_rotate(mask, _rotation, partial(drawCallback, c=maskColor, fillTextbox=fillMaskTextbox), center=_position)
                else:
                    drawCallback(pilImageDraw.Draw(mask, mode=None), maskColor, fillTextbox=fillMaskTextbox)
                mask = Convert.pil_to_cv(mask)
                mask = MaskUtil.eq_color(mask, color=maskColor)

                if maskHandler is not None and ensureNoOverlap:
                    import numpy as np
                    redo = False
                    for _mask in maskHandler:
                        # print(f"{mask.sum()=}, {_mask._mask.sum()=}")
                        mask_intersection = np.logical_and(mask, _mask._mask, dtype=np.bool_)
                        if mask_intersection.sum() > 0:
                            # print(f"{mask_intersection.sum()=}")
                            redo = True
                            break
                    if redo:
                        continue

                refMask._mask = mask

            if _rotation != 0:
                img = PilUtil._apply_rotate(img, _rotation, partial(drawCallback, c=_color), center=_position)
            else:
                drawCallback(draw, _color)
            return img

    @staticmethod
    def waku(
        img: pilImage.Image, text: StringVar,
        wakuWidth: IntVar, wakuHeight: IntVar,
        fontPath: str,
        fontSize: IntVar,
        color: ColorVar,
        position: VectorVar | PilImageVectorCallback,
        align: Literal['left', 'center', 'right']='left',
        direction: Literal['rtl', 'ltr', 'ttb']='ltr',
        rotation: FloatVar=0,
        ensureNoOverlap: bool=False,
        ensureInBounds: bool=False,
        fillMaskTextbox: bool=False,
        refMask: Mask=None,
        maskHandler: MaskHandler=None
    ) -> pilImage.Image:
        while True:
            _text = Convert.cast_str(text)
            _wakuWidth = Convert.cast_builtin(wakuWidth, asInt=True)
            _wakuHeight = Convert.cast_builtin(wakuWidth, asInt=True)

            _fontSize = Convert.cast_builtin(fontSize, asInt=True)
            _color = Convert.cast_color(color, tupleAttr='rgb', asInt=True)
            if callable(position):
                _position = position(img)
            else:
                _position = position
            _position = Convert.cast_vector(_position)
            _rotation = Convert.cast_builtin(rotation)

            draw = pilImageDraw.Draw(img, mode=None)

            def wakuDrawCallback(d: pilImageDraw.ImageDraw, c: tuple[int, int, int], fillTextbox: bool=False):
                p0 = (_position[0] - _wakuWidth/2, _position[1] - _wakuHeight/2)
                p1 = (_position[0] + _wakuWidth/2, _position[1] + _wakuHeight/2)
                shape: list[float] = list(p0) + list(p1)
                d.rectangle(xy=shape, fill=c)

            if ensureInBounds:
                if _position[0] < 0:
                    continue
                if _position[1] < 0:
                    continue
                if _position[0] + _wakuWidth > img.width:
                    continue
                if _position[1] + _wakuHeight > img.height:
                    continue

            font = pilImageFont.truetype(font=fontPath, size=_fontSize)
            w, h = draw.textsize(_text, font=font, direction=direction)
            _textPosition = Interval[Vector[int]](
                Vector[int](_position[0] + w/2, _position[1] + h/2),
                Vector[int](_position[0] + _wakuWidth - w/2, _position[1] + _wakuHeight - h/2)
            )
            _textPosition = Convert.cast_vector(_textPosition)
            xy = (_textPosition[0] - w/2, _textPosition[1] - h/2)

            def textDrawCallback(d: pilImageDraw.ImageDraw, c: tuple[int, int, int]):
                d.text(
                    xy=xy,
                    text=_text,
                    fill=c,
                    font=font,
                    align=align,
                    direction=direction,
                )
            
            def drawCallback(d: pilImageDraw.ImageDraw, c: tuple[int, int, int], fillTextbox: bool=False):
                wakuDrawCallback(d=d, c=c, fillTextbox=fillMaskTextbox)
                if not fillMaskTextbox:
                    textDrawCallback(d=d, c=c)

            if refMask is not None:
                mask = pilImage.new("RGB", (img.width, img.height), color=(0, 0, 0))
                maskColor = (255, 255, 255)
                if _rotation != 0:
                    mask = PilUtil._apply_rotate(mask, _rotation, partial(drawCallback, c=maskColor, fillTextbox=fillMaskTextbox), center=_position)
                else:
                    drawCallback(pilImageDraw.Draw(mask, mode=None), maskColor, fillTextbox=fillMaskTextbox)
                mask = Convert.pil_to_cv(mask)
                mask = MaskUtil.eq_color(mask, color=maskColor)

                if maskHandler is not None and ensureNoOverlap:
                    import numpy as np
                    redo = False
                    for _mask in maskHandler:
                        # print(f"{mask.sum()=}, {_mask._mask.sum()=}")
                        mask_intersection = np.logical_and(mask, _mask._mask, dtype=np.bool_)
                        if mask_intersection.sum() > 0:
                            # print(f"{mask_intersection.sum()=}")
                            redo = True
                            break
                    if redo:
                        continue

                refMask._mask = mask

            if _rotation != 0:
                img = PilUtil._apply_rotate(img, _rotation, partial(drawCallback, c=_color), center=_position)
            else:
                drawCallback(draw, _color)
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
        fillMaskTextbox: bool=False,
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

        def textDrawCallback(d: pilImageDraw.ImageDraw, fill: tuple, fillTextbox: bool=False):
            if not fillTextbox:
                d.text(
                    xy=textPosition, text=text, fill=fill,
                    font=font, direction=direction
                )
            else:
                p0 = (position[0] - w/2, position[1] - h/2)
                p1 = (position[0] + w/2, position[1] + h/2)
                shape: list[float] = list(p0) + list(p1)
                d.rectangle(xy=shape, fill=fill)

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
                    partial(textDrawCallback, fill=textMaskColor, fillTextbox=fillMaskTextbox),
                    center=position
                )
            else:
                textDrawCallback(pilImageDraw.Draw(textMask, mode=None), fill=textMaskColor, fillTextbox=fillMaskTextbox)
            textMask = Convert.pil_to_cv(textMask)
            textMask = MaskUtil.eq_color(textMask, color=textMaskColor)
            assert refMask._mask is not None
            refMask._mask |= textMask
            
        return img

    @staticmethod
    def bbox_text(
        img: pilImage.Image, text: StringVar, bbox: BBox2D,
        fontPath: str,
        color: ColorVar,
        side: str='top',
        direction: str='rtl',
        targetProp: float=0.8, targetDim: str='width',
        widthPropBounds: tuple(float, float)=None,
        heightPropBounds: tuple(float, float)=None,
        rotation: FloatVar=0,
        fillMaskTextbox: bool=False,
        drawBbox: bool=False,
        drawTextBbox: bool=False,
        bboxLineWidth: float=1,
        refMask: Mask=None
    ):
        baseFontSize = 20

        text = Convert.cast_str(text)
        color = Convert.cast_color(color, tupleAttr='rgb', asInt=True)
        bboxWidth = bbox.xInterval.length
        bboxHeight = bbox.yInterval.length
        if side in ['top', 't']:
            anchor = 'mb'
            p0 = Vector2(bbox.v1.x, bbox.v0.y)
            p1 = bbox.v0
            refPoint = 0.5 * (p0 + p1)
            offsetDir = Vector2.down
        elif side in ['bottom', 'b']:
            anchor = 'mt'
            p0 = Vector2(bbox.v0.x, bbox.v1.y)
            p1 = bbox.v1
            refPoint = 0.5 * (p0 + p1)
            offsetDir = Vector2.up
        elif side in ['left', 'l']:
            anchor = 'rm'
            p0 = Vector2(bbox.v0.x, bbox.v1.y)
            p1 = bbox.v0
            refPoint = 0.5 * (p0 + p1)
            offsetDir = Vector2.left
        elif side in ['right', 'r']:
            anchor = 'lm'
            p0 = Vector2(bbox.v1.x, bbox.v0.y)
            p1 = bbox.v1
            refPoint = 0.5 * (p0 + p1)
            offsetDir = Vector2.right
        else:
            raise ValueError
        
        draw = pilImageDraw.Draw(img, mode=None)
        font = pilImageFont.truetype(font=fontPath, size=baseFontSize)
        w, h = draw.textsize(text, font=font, direction=direction)
        wProp = w / bboxWidth; hProp = h / bboxHeight

        def adjustScaleToWidthBounds(scale: float) -> float:
            if widthPropBounds is not None:
                minWidthProp, maxWidthProp = widthPropBounds
                if scale * wProp > maxWidthProp:
                    scale = maxWidthProp / wProp
                if scale * wProp < minWidthProp:
                    scale = minWidthProp / wProp
            return scale

        def adjustScaleToHeightBounds(scale: float) -> float:
            if heightPropBounds is not None:
                minHeightProp, maxHeightProp = heightPropBounds
                if scale * hProp > maxHeightProp:
                    scale = maxHeightProp / hProp
                if scale * hProp < minHeightProp:
                    scale = minHeightProp / hProp
            return scale

        if targetDim == 'width':
            targetScale = targetProp / wProp
            targetScale = adjustScaleToHeightBounds(targetScale)
            targetScale = adjustScaleToWidthBounds(targetScale)
        elif targetDim == 'height':
            targetScale = targetProp / hProp
            targetScale = adjustScaleToWidthBounds(targetScale)
            targetScale = adjustScaleToHeightBounds(targetScale)
        else:
            raise ValueError

        targetFontSize = targetScale * baseFontSize
        targetFontSize = max(1, int(targetFontSize))
        targetFont = pilImageFont.truetype(font=fontPath, size=targetFontSize)
        targetW, targetH = draw.textsize(text, font=targetFont, direction=direction)
        position = refPoint + abs(Vector2.Dot(Vector2(targetW/2, targetH/2), offsetDir)) * offsetDir
        textPosition = refPoint
        position = tuple(position)
        textPosition = tuple(textPosition)

        def textDrawCallback(d: pilImageDraw.ImageDraw, fill: tuple, fillTextbox: bool=False):
            if not fillTextbox:
                d.text(
                    xy=textPosition, text=text, fill=fill,
                    font=targetFont, direction=direction,
                    anchor=anchor
                )
                if drawTextBbox:
                    p0 = (position[0] - targetW/2, position[1] - targetH/2)
                    p1 = (position[0] + targetW/2, position[1] + targetH/2)
                    shape: list[float] = list(p0) + list(p1)
                    d.rectangle(xy=shape, outline=fill, width=bboxLineWidth)
            else:
                p0 = (position[0] - targetW/2, position[1] - targetH/2)
                p1 = (position[0] + targetW/2, position[1] + targetH/2)
                shape: list[float] = list(p0) + list(p1)
                d.rectangle(xy=shape, fill=fill)

        def bboxDrawCallback(d: pilImageDraw.ImageDraw, outline: tuple):
            p0 = tuple(bbox.v0)
            p1 = tuple(bbox.v1)
            shape: list[float] = list(p0) + list(p1)
            d.rectangle(xy=shape, outline=outline, width=bboxLineWidth)

        if rotation != 0:
            img = PilUtil._apply_rotate(
                img, rotation,
                partial(textDrawCallback, fill=color),
                center=position
            )
            if drawBbox:
                img = PilUtil._apply_rotate(
                    img, rotation,
                    partial(bboxDrawCallback, outline=color)
                )
        else:
            textDrawCallback(draw, fill=color)
            if drawBbox:
                bboxDrawCallback(draw, outline=color)

        if refMask is not None:
            textMask = pilImage.new("RGB", (img.width, img.height), color=(0, 0, 0))
            textMaskColor = (255, 255, 255)
            if rotation != 0:
                textMask = PilUtil._apply_rotate(
                    textMask, rotation,
                    partial(textDrawCallback, fill=textMaskColor, fillTextbox=fillMaskTextbox),
                    center=position
                )
                if drawBbox:
                    img = PilUtil._apply_rotate(
                        textMask, rotation,
                        partial(bboxDrawCallback, outline=textMaskColor)
                    )
            else:
                textDrawCallback(pilImageDraw.Draw(textMask, mode=None), fill=textMaskColor, fillTextbox=fillMaskTextbox)
                if drawBbox:
                    bboxDrawCallback(pilImageDraw.Draw(textMask, mode=None), outline=textMaskColor)
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
        PilUtil.hanko(
            img, text="夏生", fontPath=PilUtil.defaultFontPath, fontSize=150, color=(255, 0, 0),
            position=(250, 250), direction='ttb', outlineWidthRatio=20,
            marginOffset=0, marginRatio=0.1
        )
        img.show()
    
    @staticmethod
    def debug_text():
        img = pilImage.new("RGB", (500, 500), color=(0, 0, 255))
        img = PilUtil.text(
            img=img, _text='hello', fontPath=PilUtil.defaultFontPath, _fontSize=150,
            _color=(255, 0, 0), _position=(250, 250), _rotation=45
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

    @staticmethod
    def debug_bbox_text():
        img = pilImage.new("RGB", (500, 500), color=(0, 0, 255))

        img = PilUtil.bbox_text(
            img=img, text='物凄く長い文字列',
            bbox=BBox2D(Vector2(200,200), Vector2(300,300)),
            fontPath=PilUtil.defaultFontPath,
            color=(0,255,0),
            # side='top', direction='rtl', targetProp=0.8, targetDim='width',
            side='left', direction='ttb', targetProp=0.8, targetDim='height',

            rotation=0,
            fillMaskTextbox=False,
            drawBbox=True,
            drawTextBbox=True,
            bboxLineWidth=1,
            refMask=None
        )
        img.show()