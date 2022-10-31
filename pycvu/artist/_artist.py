from __future__ import annotations
from functools import partial
import json
from typing import Callable, cast, overload
import cv2
import numpy as np
from pyevu import Vector2
from ..util import CvUtil

__all__ = [
    "Artist"
]

"""
TODO: Need to implement methods for drawing shapes and text randomly.
"""

import random
from typing import Generic, TypeVar
T = TypeVar('T', int, float)

from ..base import Base, BaseHandler

class Vector(Vector2, Base, Generic[T]):
    def __init__(self, x: T, y: T):
        assert type(x) is type(y)
        assert type(x) in T.__constraints__
        Vector2.__init__(self, x=x, y=y)
        Base.__init__(self)
    
    def __str__(self) -> str:
        return f"{type(self).__name__}[{self.generic_type.__name__}]({','.join([f'{key}={val}' for key, val in self.__dict__.items() if key != '__orig_class__'])})"

    @property
    def generic_type(self) -> type: # Note: Can't be called from __init__ or any other dunder method.
        return self.__orig_class__.__args__[0]

    @staticmethod
    def debug():
        intVec = Vector[int](1, 2)
        print(f"{intVec=}")
        print(f"{intVec.to_dict()=}")
        print(f"{Vector[int].from_dict(intVec.to_dict())=}")
        intVecCopy = Vector[int].from_dict(intVec.to_dict())

class IntVectorList(BaseHandler[Vector[int]]):
    def __init__(self, _objects: list[Vector[int]]=None):
        super().__init__(_objects)
    
    @staticmethod
    def debug():
        vectors = IntVectorList([
            Vector[int](1, 2), Vector[int](3, 4),
            Vector[int](5, 6), Vector[int](7, 8),
        ])
                
        print(f"{vectors.to_dict()=}")
        print(f"{IntVectorList.from_dict(vectors.to_dict())=}")
        assert vectors.to_dict() == IntVectorList.from_dict(vectors.to_dict()).to_dict()
        assert vectors == IntVectorList.from_dict(vectors.to_dict())
        print('Test passed')

class GenericVectorList(BaseHandler[Vector[T]]):
    def __init__(self, _objects: list[Vector[T]]=None):
        super().__init__(_objects)

    @staticmethod
    def debug():
        vectors = GenericVectorList[int]([
            Vector[int](1, 2), Vector[int](3, 4),
            Vector[int](5, 6), Vector[int](7, 8),
        ])                
        print(f"{vectors.to_dict()=}")
        print(f"{GenericVectorList[int].from_dict(vectors.to_dict()).to_dict()=}")
        assert vectors.to_dict() == GenericVectorList[int].from_dict(vectors.to_dict()).to_dict()
        assert vectors == GenericVectorList[int].from_dict(vectors.to_dict())
        print('Test passed')

TV = TypeVar('TV', int, float, Vector)

class Interval(Base, Generic[TV]):
    def __init__(self, valMin: TV, valMax: TV):
        super().__init__()
        assert type(valMin) is type(valMax)
        assert type(valMin) in TV.__constraints__
        if type(valMin) in T.__constraints__:
            assert valMin <= valMax
        elif type(valMin) is Vector and type(valMax) is Vector:
            assert valMin.generic_type is valMax.generic_type
            assert valMin.x <= valMax.x and valMin.y <= valMax.y
        else:
            raise TypeError

        self.valMin = valMin
        self.valMax = valMax

    @property
    def generic_type(self) -> type: # Note: Can't be called from __init__
        return self.__orig_class__.__args__[0]

    def random(self) -> TV:
        if self.generic_type is float:
            randVal = random.random()
            return self.valMin + (self.valMax - self.valMin) * randVal
        elif self.generic_type is int:
            return random.randint(self.valMin, self.valMax)
        elif self.generic_type in [Vector[constraintType] for constraintType in T.__constraints__]:
            genType = self.valMin.generic_type
            x = Interval[genType](self.valMin.x, self.valMax.x).random()
            y = Interval[genType](self.valMin.y, self.valMax.y).random()
            return Vector[genType](x, y)
        else:
            raise TypeError
    
    @staticmethod
    def debug():
        print("Random Integer Vector Interval")
        intVecInterval = Interval[Vector[int]](Vector[int](1, 2), Vector[int](10, 20))
        for i in range(10):
            print(f"\t{i}: {intVecInterval.random()=}")
        print("Random Integer Vector Interval")
        floatVecInterval = Interval[Vector[float]](Vector[float](1.0, 2.0), Vector[float](10.0, 20.0))
        for i in range(10):
            print(f"\t{i}: {floatVecInterval.random()=}")
        
        print(f"{dir(intVecInterval)=}")
        print(f"{intVecInterval.__dict__=}")
        hasToDict = {key: hasattr(val, 'to_dict') for key, val in intVecInterval.__dict__.items()}
        print(f"Has to_dict: {hasToDict}")
        print(f"{intVecInterval.to_dict()=}")


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
