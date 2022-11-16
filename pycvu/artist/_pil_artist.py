from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Literal

import pycvu

from ..util import Convert, PilUtil, \
    VectorVar, PilImageVectorCallback, \
    IntVar, FloatVar, StringVar
if TYPE_CHECKING:
    from ._artist import Artist
from ._draw_process import DrawProcess
from ._img_type import ImageType

__all__ = [
    "PilArtist"
]

class PilArtist:
    fontSize: IntVar = 30
    """Size of the font when drawing text."""

    fontPath: str = f"{pycvu.__path__[0]}/data/font/ipaexg.ttf"
    """Path to the font used when drawing text."""

    fillMaskTextbox: bool = True
    """When drawing the mask for text, draw a rectangle instead of the actual text."""

    hankoIsVertical: bool = True
    hankoOutlineWidthRatio: FloatVar = 0.1
    hankoMarginOffset: FloatVar = 0
    hankoMarginRatio: FloatVar = 0

    def __init__(self, artist: Artist):
        self._artist = artist

    def text(
        self, text: StringVar,
        position: VectorVar | PilImageVectorCallback,
        direction: Literal['rtl', 'ltr', 'ttb']='ltr',
        rotation: FloatVar=0,
        weight: float=1, repeat: int=1, prob: float=1.0
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
            rotation=rotation,
            fillMaskTextbox=PilArtist.fillMaskTextbox
        )
        maskSetting = self._artist.maskSetting.copy() if not self._artist.maskSetting.skip else None
        self._artist._add_process(DrawProcess(p, maskSetting, weight=weight, repeat=repeat, prob=prob, imgType=ImageType.PIL))
        return self
    
    def hanko(
        self, text: StringVar,
        position: VectorVar | PilImageVectorCallback,
        rotation: FloatVar=0,
        weight: float=1, repeat: int=1, prob: float=1.0
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
            rotation=rotation,
            fillMaskTextbox=PilArtist.fillMaskTextbox
        )
        maskSetting = self._artist.maskSetting.copy() if not self._artist.maskSetting.skip else None
        self._artist._add_process(DrawProcess(p, maskSetting, weight=weight, repeat=repeat, prob=prob, imgType=ImageType.PIL))
        return self
