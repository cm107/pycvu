from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
from ..vector import Vector
import pycvu

from ..util import Util, PilUtil
if TYPE_CHECKING:
    from ._artist import Artist

__all__ = [
    "PilArtist"
]

class PilArtist:
    fontSize: int = 30
    """Size of the font when drawing text."""

    fontPath: str = f"{pycvu.__path__[0]}/data/font/ipaexg.ttf"
    """Path to the font used when drawing text."""

    hankoIsVertical: bool = True
    hankoOutlineWidth: float = 20
    hankoMarginOffset: float = 0
    hankoMarginRatio: float = 0

    def __init__(self, artist: Artist):
        self._artist = artist
    
    @staticmethod
    def _pillow_decorator(method):
        def _inner(ref: PilArtist, *args, **kwargs):
            ref._artist._drawQueue.append(
                partial(Util.cv_to_pil)
            )
            ref = method(ref, *args, **kwargs)
            ref._artist._drawQueue.append(
                partial(Util.pil_to_cv)
            )
            return ref
        return _inner

    @_pillow_decorator
    def text(
        self, text: str,
        position: tuple[float, float] | Vector
    ) -> Artist:
        """Draws text on the image.

        Args:
            text (str): The text that you would like to draw.
            position (tuple[float, float]): Where you would like to draw the text.
        """
        self._artist._drawQueue.append(
            partial(
                PilUtil.text,
                text=text,
                fontPath=PilArtist.fontPath,
                fontSize=PilArtist.fontSize,
                color=Util.bgr_to_rgb(self._artist.color),
                position=position
            )
        )
        return self
    
    @_pillow_decorator
    def hanko(
        self, text: str, position: tuple[float, float] | Vector
    ) -> Artist:
        self._artist._drawQueue.append(
            partial(
                PilUtil.hanko,
                text=text,
                fontPath=PilArtist.fontPath, fontSize=PilArtist.fontSize,
                color=self._artist.color,
                position=position,
                direction='ttb' if PilArtist.hankoIsVertical else 'ltr',
                outlineWidth=PilArtist.hankoOutlineWidth,
                marginOffset=PilArtist.hankoMarginOffset,
                marginRatio=PilArtist.hankoMarginRatio
            )
        )
        return self
