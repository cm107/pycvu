from __future__ import annotations
from functools import partial
import importlib
from typing import Callable, overload
import types
import cv2
import numpy as np

from pycvu.base import BaseUtil, Base

from ..color import Color
from ..mask import MaskSetting, Mask, MaskHandler
from ..vector import Vector
from ..util import CvUtil, \
    VectorVar, ImageVectorCallback, ColorVar, \
    IntVar, FloatVar

__all__ = [
    "Artist"
]

"""
* Need to implement logic to keep track of masks. Refer to Kume's algorithms.
    * At the very least, I need to get the mask of each hanko for calculating the bbox and segmentation.
    * It is possible that the hanko will have occlusion if something else is drawn on top of it.
    * Do I need to generate a mask for each thing that is drawn?
        * That would be the most flexible, but the hardest to implement.
        * It might also take up considerable memory at runtime?
        * Doing this would make it easy to account for occlusion.
            * I could step through each mask and erase the parts that overlap with proceeding masks.
        * Resize and rotation operations would affect the masks as well.
            * This needs to be accounted for with caution.
            * I'm guessing the interpolation method is going to matter when resizing/rotating a boolean mask.
    * I should probably pass a list of masks into each drawing method by reference.
        * This way the masks can be updated whenever necessary.
        * It can be an optional parameter.
        * Approach Possibilities:
            * Keep track of the mask of all drawing operations.
                * This would take up more memory at runtime.
            * Keep track of just the masks that matter.
                * This would take up less memory at runtime.
                * Masks would need to be updated on-the-go, since the other masks still cause occlusion.
"""

from ..interval import Interval
from ..vector import Vector


class Artist(Base):
    color: ColorVar = Color(255, 255, 255)
    """Color used when drawing anything."""

    thickness: IntVar = 1
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
    
    fontScale: FloatVar = 1.0
    """Font scale used when drawing text."""

    maskSetting: MaskSetting = MaskSetting()
    """Use for controlling which masks should be tracked and/or contribute to occlusion."""

    from ._pil_artist import PilArtist as PIL

    def __init__(self, src: np.ndarray | str):
        super().__init__()
        if type(src) is np.ndarray:
            img = src
        elif type(src) is str:
            img = cv2.imread(src)
        else:
            raise TypeError
        self._img = img
        self._drawQueue: list[Callable[[np.ndarray], np.ndarray]] = []
        self._maskSettingDict: dict[int, MaskSetting] = {}

        self.pil = Artist.PIL(self)
    
    def to_dict(self, saveImg: bool=True, saveMeta: bool=True) -> dict:
        itemDict: dict = dict()
        itemDict['_img'] = self._img.tolist() if saveImg else None
        itemDict['_drawQueue'] = self._serialize_queue()
        itemDict['_maskSettingDict'] = {idx: maskSetting.to_dict() for idx, maskSetting in self._maskSettingDict.items()}
        
        def get_meta_dict(objCls, excludedKeys: list[str]=[]) -> dict:
            metaDict: dict = dict()
            for key, val in objCls.__dict__.items():
                if (
                    key.startswith('_') or key in excludedKeys
                    or type(val) in [
                        types.FunctionType,
                        classmethod,
                        staticmethod
                    ]
                ):
                    continue
                if hasattr(val, 'to_dict'):
                    metaDict[key] = val.to_dict()
                else:
                    metaDict[key] = val
            return metaDict
        
        meta: dict = dict(
            cv=get_meta_dict(self.__class__, excludedKeys=['PIL']),
            pil=get_meta_dict(self.pil.__class__)
        ) if saveMeta else None
        itemDict['meta'] = meta
        return itemDict

    @classmethod
    def from_dict(cls, item_dict: dict, img: np.ndarray=None, loadMeta: bool=True) -> Artist:
        if img is None and item_dict['_img'] is None:
            raise ValueError("Image data not found in item_dict. Must specify img parameter.")
        elif img is None:
            img = np.array(item_dict['_img'], dtype=np.uint8)
        assert img is not None
        artist = Artist(img)
        
        def set_meta(metaObjCls, metaDict: dict):
            for key, val in metaDict.items():
                assert hasattr(metaObjCls, key)
                if type(val) is dict and '_typedict' in val:
                    objCls = BaseUtil.from_type_dict(val['_typedict'])
                    if hasattr(objCls, 'from_dict'):
                        obj = objCls.from_dict(val)
                    else:
                        obj = objCls(**{key0: val0 for key0, val0 in val.items() if key0 != '_typedict'})
                    setattr(metaObjCls, key, obj)
                else:
                    setattr(metaObjCls, key, val)
        
        if loadMeta and item_dict['meta'] is not None:
            set_meta(artist.__class__, item_dict['meta']['cv'])
            set_meta(artist.pil.__class__, item_dict['meta']['pil'])
        
        artist._drawQueue = artist._unserialize_queue(item_dict['_drawQueue'])
        artist._maskSettingDict = {int(idx): MaskSetting.from_dict(maskSetting) for idx, maskSetting in item_dict['_maskSettingDict'].items()}
        return artist

    def save(self, path: str, saveImg: bool=True, saveMeta: bool=True):
        super().save(path, saveImg=saveImg, saveMeta=saveMeta)

    @classmethod
    def load(cls, path: str, img: np.ndarray=None, loadMeta: bool=True) -> Artist:
        return super().load(path, img=img, loadMeta=loadMeta)

    def circle(
        self, center: VectorVar | ImageVectorCallback,
        radius: IntVar, fill: bool=False
    ) -> Artist:
        """Draws a circle.

        Args:
            center (tuple[int, int] | Vector): Center of the circle.
            radius (int): Radius of the circle.
            fill (bool, optional): Whether or not to fill the circle in. Defaults to False.
        """
        if not self.maskSetting.skip:
            self._maskSettingDict[len(self._drawQueue)] = self.maskSetting.copy()
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
        self, center: VectorVar | ImageVectorCallback,
        axis: VectorVar,
        angle: FloatVar=0,
        startAngle: FloatVar=0,
        endAngle: FloatVar=360,
        fill: bool=False
    ) -> Artist:
        """Draws an ellipse.

        Args:
            center (tuple[int, int] | Vector): Center of the ellipse.
            axis (tuple[int, int] | Vector): The a and b axes of the ellipse. Major and minor axis is not in any particular order.
            angle (float, optional): Angle by which the ellipse is rotated before drawing. Defaults to 0.
            startAngle (float, optional): The angle that you want to start drawing the ellipse at. Defaults to 0.
            endAngle (float, optional): The angle that you want to stop drawing the ellipse at. Defaults to 360.
            fill (bool, optional): Whether or not you want to fill in the ellipse when drawing. Defaults to False.
        """
        if not self.maskSetting.skip:
            self._maskSettingDict[len(self._drawQueue)] = self.maskSetting.copy()
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

    def rectangle(
        self,
        pt1: VectorVar | ImageVectorCallback,
        pt2: VectorVar | ImageVectorCallback,
        fill: bool=False
    ) -> Artist:
        """Draws a rectangle.

        Args:
            pt1 (tuple[int, int] | Vector): First corner of rectangle.
            pt2 (tuple[int, int] | Vector): Second corner of rectangle.
            fill (bool, optional): Whether or not to fill in the rectangle when drawing. Defaults to False.
        """
        if not self.maskSetting.skip:
            self._maskSettingDict[len(self._drawQueue)] = self.maskSetting.copy()
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
    
    def line(
        self,
        pt1: VectorVar | ImageVectorCallback,
        pt2: VectorVar | ImageVectorCallback
    ) -> Artist:
        """Draws a line.

        Args:
            pt1 (tuple[int, int] | Vector): Where to start drawing the line.
            pt2 (tuple[int, int] | Vector): Where to stop drawing the line.
        """
        if not self.maskSetting.skip:
            self._maskSettingDict[len(self._drawQueue)] = self.maskSetting.copy()
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
        self, angle: FloatVar,
        degrees: bool=True,
        scale: FloatVar=1,
        adjustBorder: bool=False,
        center: VectorVar=None
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
            center (tuple[int, int] | Vector, optional): The center of rotation. Defaults to center of the image.
        """
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
        self, text: str,
        org: VectorVar,
        bottomLeftOrigin: bool=False
    ) -> Artist:
        """Draws text on the image.

        Args:
            text (str): The text string that you would like to draw.
            org (tuple[int, int] | Vector): Where the text should be drawn.
            bottomLeftOrigin (bool, optional):
                In OpenCV coordinates, (0,0) is assumed to be at the upper-left
                corner of the image. Setting bottomLeftOrigin=True changes the
                assumption to be the bottom-left corner of the image.
                Defaults to False.
        """
        if not self.maskSetting.skip:
            self._maskSettingDict[len(self._drawQueue)] = self.maskSetting.copy()
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
    def resize(self, dsize: VectorVar) -> Artist:
        """Resizes the working image to the given target size.

        Args:
            dsize (tuple[int, int] | Vector): target size
        """
        ...

    @overload
    def resize(self, fx: FloatVar=None, fy: FloatVar=None) -> Artist:
        """Resizes the working image according to the given scaling factors.
        Scaling factors are relative to the current working image size.

        Args:
            fx (float, optional): x resize scaling factor. Defaults to None.
            fy (float, optional): y resize scaling factor. Defaults to None.
        """
        ...

    def resize(
        self, dsize: VectorVar=None,
        fx: FloatVar=None, fy: FloatVar=None
    ) -> Artist:
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
    
    def draw_and_get_masks(self) -> tuple[np.ndarray, MaskHandler]:
        """Perform all of the actions in the drawing queue on the source image and return the result.
        """
        import inspect
        result = self._img.copy()

        maskHandler = MaskHandler()
        for i, hook in enumerate(self._drawQueue):
            funcArgs: list[str] = inspect.getfullargspec(hook.func).args
            if 'refMask' in funcArgs and i in self._maskSettingDict:
                mask = Mask(setting=self._maskSettingDict[i])
                result = hook(result, refMask=mask)
                maskHandler.process(mask)
            elif 'maskHandler' in funcArgs:
                result = hook(result, maskHandler=maskHandler)
            elif i in self._maskSettingDict:
                print(f"TODO: Implement refMask for {hook.func.__qualname__}")
                result = hook(result)
            else:
                result = hook(result)
        return result, maskHandler

    def draw_and_save(self, path: str):
        """Perform all of the actions in the drawing queue on the source image and save the result to a file.
        """
        cv2.imwrite(path, self.draw())

    def _serialize_queue(self) -> list[dict]:
        result: list[dict] = []
        for p in self._drawQueue:
            result.append(BaseUtil.to_func_dict(p))
        return result

    def _unserialize_queue(self, serializedQueue: list[dict]) -> list[Callable[[np.ndarray], np.ndarray]]:
        queue: list[Callable[[np.ndarray], np.ndarray]] = []
        for pDict in serializedQueue:
            queue.append(BaseUtil.from_func_dict(pDict))
        return queue

    from ._debug import debug
