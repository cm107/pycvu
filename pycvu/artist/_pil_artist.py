from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Literal

import pycvu

from ..util import Convert, PilUtil, \
    VectorVar, PilImageVectorCallback, \
    IntVar, FloatVar, StringVar, \
    RepeatDrawCallback
if TYPE_CHECKING:
    from ._artist import Artist
from ._draw_process import DrawProcess

__all__ = [
    "PilArtist"
]

class PilArtist:
    fontSize: IntVar = 30
    """Size of the font when drawing text."""

    fontPath: str = f"{pycvu.__path__[0]}/data/font/ipaexg.ttf"
    """Path to the font used when drawing text."""

    hankoIsVertical: bool = True
    hankoOutlineWidthRatio: FloatVar = 0.1
    hankoMarginOffset: FloatVar = 0
    hankoMarginRatio: FloatVar = 0

    def __init__(self, artist: Artist):
        self._artist = artist
    
    @staticmethod
    def _pillow_decorator(method):
        def _inner(ref: PilArtist, *args, **kwargs):
            ref._artist._drawQueue.append(
                DrawProcess(partial(Convert.cv_to_pil))
            )
            ref = method(ref, *args, **kwargs)
            ref._artist._drawQueue.append(
                DrawProcess(partial(Convert.pil_to_cv))
            )
            return ref
        return _inner

    @_pillow_decorator
    def text(
        self, text: StringVar,
        position: VectorVar | PilImageVectorCallback,
        direction: Literal['rtl', 'ltr', 'ttb']='ltr',
        rotation: FloatVar=0,
        repeat: int=1
    ) -> Artist:
        """Draws text on the image.

        Args:
            text (str): The text that you would like to draw.
            position (tuple[float, float]): Where you would like to draw the text.
        """
        p = partial(
            PilUtil.text,
            text=text,
            fontPath=PilArtist.fontPath,
            fontSize=PilArtist.fontSize,
            color=Convert.bgr_to_rgb(self._artist.color),
            position=position,
            direction=direction,
            rotation=rotation
        )
        if repeat > 1:
            p = RepeatDrawCallback(p, repeat=repeat)
        maskSetting = self._artist.maskSetting.copy() if not self._artist.maskSetting.skip else None
        self._artist._drawQueue.append(DrawProcess(p, maskSetting))
        return self
    
    @_pillow_decorator
    def hanko(
        self, text: StringVar,
        position: VectorVar | PilImageVectorCallback,
        rotation: FloatVar=0,
        repeat: int=1
    ) -> Artist:
        p = partial(
            PilUtil.hanko,
            text=text,
            fontPath=PilArtist.fontPath, fontSize=PilArtist.fontSize,
            color=Convert.bgr_to_rgb(self._artist.color),
            position=position,
            direction='ttb' if PilArtist.hankoIsVertical else 'ltr',
            outlineWidthRatio=PilArtist.hankoOutlineWidthRatio,
            marginOffset=PilArtist.hankoMarginOffset,
            marginRatio=PilArtist.hankoMarginRatio,
            rotation=rotation
        )
        if repeat > 1:
            p = RepeatDrawCallback(p, repeat=repeat)
        maskSetting = self._artist.maskSetting.copy() if not self._artist.maskSetting.skip else None
        self._artist._drawQueue.append(DrawProcess(p, maskSetting))
        return self
