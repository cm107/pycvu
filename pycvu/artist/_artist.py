from __future__ import annotations
from functools import partial
from typing import Callable, overload
import cv2
import numpy as np
from pyevu import Vector2
from ..util import CvUtil

__all__ = [
    "Artist"
]

class Artist:
    color: tuple[int, int, int] = (255, 255, 255)
    """Color used when drawing anything."""

    thickness: int = 1
    """Line thickness used when drawing non-solid shapes."""

    lineType: int = cv2.LINE_AA
    """Line type used when drawing non-solid shapes."""

    interpolation: int = cv2.INTER_AREA
    """
    Interpolation used when resizing.
    This can be any cv2.INTER_* value.
    """

    fontFace: int = cv2.FONT_HERSHEY_COMPLEX
    """Font face usedd when drawing text."""
    
    fontScale: float = 1.0
    """Font scale used when drawing text."""

    from ._pil_artist import PilArtist as PIL

    def __init__(self, src: np.ndarray | str):
        if type(src) is np.ndarray:
            img = src
        elif type(src) is str:
            img = cv2.imread(src)
        else:
            raise TypeError
        self._img = img
        self._drawQueue: list[Callable[[np.ndarray], np.ndarray]] = []

        self.pil = Artist.PIL(self)
    
    def circle(self, center: tuple[int, int] | Vector2, radius: int, fill: bool=False) -> Artist:
        """Draws a circle.

        Args:
            center (tuple[int, int] | Vector2): Center of the circle.
            radius (int): Radius of the circle.
            fill (bool, optional): Whether or not to fill the circle in. Defaults to False.
        """
        if type(center) is Vector2:
            center = tuple(center)
        center = CvUtil.cast_int(center)
        self._drawQueue.append(
            partial(
                CvUtil.circle,
                center=center, radius=radius,
                color=Artist.color,
                thickness=Artist.thickness if not fill else -1,
                lineType=Artist.lineType
            )
        )
        return self
    
    def ellipse(
        self, center: tuple[int, int] | Vector2, axis: tuple[int, int] | Vector2,
        angle: float=0, startAngle: float=0, endAngle: float=360,
        fill: bool=False
    ) -> Artist:
        """Draws an ellipse.

        Args:
            center (tuple[int, int] | Vector2): Center of the ellipse.
            axis (tuple[int, int] | Vector2): The a and b axes of the ellipse. Major and minor axis is not in any particular order.
            angle (float, optional): Angle by which the ellipse is rotated before drawing. Defaults to 0.
            startAngle (float, optional): The angle that you want to start drawing the ellipse at. Defaults to 0.
            endAngle (float, optional): The angle that you want to stop drawing the ellipse at. Defaults to 360.
            fill (bool, optional): Whether or not you want to fill in the ellipse when drawing. Defaults to False.
        """
        if type(center) is Vector2:
            center = tuple(center)
        if type(axis) is Vector2:
            axis = tuple(axis)
        center = CvUtil.cast_int(center)
        axis = CvUtil.cast_int(axis)
        self._drawQueue.append(
            partial(
                CvUtil.ellipse,
                center=center, axis=axis,
                angle=angle, startAngle=startAngle, endAngle=endAngle,
                color=Artist.color,
                thickness=Artist.thickness if not fill else -1,
                lineType=Artist.lineType
            )
        )

    def rectangle(self, pt1: tuple[int, int] | Vector2, pt2: tuple[int, int] | Vector2, fill: bool=False) -> Artist:
        """Draws a rectangle.

        Args:
            pt1 (tuple[int, int] | Vector2): First corner of rectangle.
            pt2 (tuple[int, int] | Vector2): Second corner of rectangle.
            fill (bool, optional): Whether or not to fill in the rectangle when drawing. Defaults to False.
        """
        if type(pt1) is Vector2:
            pt1 = tuple(pt1)
        if type(pt2) is Vector2:
            pt2 = tuple(pt2)
        pt1 = CvUtil.cast_int(pt1)
        pt2 = CvUtil.cast_int(pt2)
        self._drawQueue.append(
            partial(
                CvUtil.rectangle,
                pt1=pt1, pt2=pt2,
                color=Artist.color,
                thickness=Artist.thickness if not fill else -1,
                lineType=Artist.lineType
            )
        )
        return self
    
    def line(self, pt1: tuple[int, int] | Vector2, pt2: tuple[int, int] | Vector2) -> Artist:
        """Draws a line.

        Args:
            pt1 (tuple[int, int] | Vector2): Where to start drawing the line.
            pt2 (tuple[int, int] | Vector2): Where to stop drawing the line.
        """
        if type(pt1) is Vector2:
            pt1 = tuple(pt1)
        if type(pt2) is Vector2:
            pt2 = tuple(pt2)
        pt1 = CvUtil.cast_int(pt1)
        pt2 = CvUtil.cast_int(pt2)
        self._drawQueue.append(
            partial(
                CvUtil.line,
                pt1=pt1, pt2=pt2,
                color=Artist.color,
                thickness=Artist.thickness,
                lineType=Artist.lineType
            )
        )
        return self
    
    def affine_rotate(
        self, angle: float, degrees: bool=True,
        scale: float=1, adjustBorder: bool=False,
        center: tuple[int, int] | Vector2=None
    ) -> Artist:
        """This rotates the image about the axis coming out of the screen.

        Args:
            angle (float): angle of rotation
            degrees (bool, optional): Whether or not the angle is given in degrees. Defaults to True.
            scale (float, optional):
                Scaling factor to apply to the image after being rotated.
                Without rescaling the image projection, parts of the original image may be cut off.
                Alternatively, you can use adjustBorder=True to automatically adjust the scale to fit the window perfectly.
                Defaults to 1.
            adjustBorder (bool, optional):
                If set to True, the image projection will automatically be rescaled such
                that the image fits perfectly in the window.
                Defaults to False.
            center (tuple[int, int] | Vector2, optional): The center of rotation. Defaults to center of the image.
        """
        if center is not None and type(center) is Vector2:
            center = tuple(center)
        self._drawQueue.append(
            partial(
                CvUtil.affine_rotate,
                angle=angle, degrees=degrees,
                scale=scale, interpolation=Artist.interpolation,
                adjustBorder=adjustBorder, center=center,
                borderColor=Artist.color
            )
        )
        return self

    def text(
        self, text: str, org: tuple[int, int] | Vector2,
        bottomLeftOrigin: bool=False
    ) -> Artist:
        """Draws text on the image.

        Args:
            text (str): The text string that you would like to draw.
            org (tuple[int, int] | Vector2): Where the text should be drawn.
            bottomLeftOrigin (bool, optional):
                In OpenCV coordinates, (0,0) is assumed to be at the upper-left
                corner of the image. Setting bottomLeftOrigin=True changes the
                assumption to be the bottom-left corner of the image.
                Defaults to False.
        """
        if type(org) is Vector2:
            org = tuple(org)
        org = CvUtil.cast_int(org)
        self._drawQueue.append(
            partial(
                CvUtil.text,
                text=text, org=org,
                fontFace=Artist.fontFace, fontScale=Artist.fontScale,
                color=Artist.color, thickness=Artist.thickness,
                lineType=Artist.lineType,
                bottomLeftOrigin=bottomLeftOrigin
            )
        )

    @overload
    def resize(self, dsize: tuple[int, int] | Vector2) -> Artist:
        """Resizes the working image to the given target size.

        Args:
            dsize (tuple[int, int] | Vector2): target size
        """
        ...

    @overload
    def resize(self, fx: float=None, fy: float=None) -> Artist:
        """Resizes the working image according to the given scaling factors.
        Scaling factors are relative to the current working image size.

        Args:
            fx (float, optional): x resize scaling factor. Defaults to None.
            fy (float, optional): y resize scaling factor. Defaults to None.
        """
        ...

    def resize(
        self, dsize: tuple[int, int] | Vector2=None,
        fx: float=None, fy: float=None
    ) -> Artist:
        if dsize is not None:
            if type(dsize) is Vector2:
                dsize = tuple(Vector2)
            dsize = CvUtil.cast_int(dsize)
        self._drawQueue.append(
            partial(
                CvUtil.resize,
                dsize=dsize,
                fx=fx, fy=fy,
                interpolation=Artist.interpolation
            )
        )

    def draw(self) -> np.ndarray:
        """Perform all of the actions in the drawing queue on the source image and return the result.
        """
        result = self._img.copy()
        for hook in self._drawQueue:
            result = hook(result)
        return result
    
    def save(self, path: str):
        """Perform all of the actions in the drawing queue on the source image and save the result to a file.
        """
        cv2.imwrite(path, self.draw())

    from ._debug import debug
