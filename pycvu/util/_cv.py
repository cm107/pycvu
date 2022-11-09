from __future__ import annotations
import math
from typing import overload, TYPE_CHECKING, Callable
from functools import partial
import cv2
import numpy as np
import numpy.typing as npt
from pyevu import BBox2D, Vector2

from pycvu.interval import Interval

from ..color import Color
from ..vector import Vector
if TYPE_CHECKING:
    from ..mask import Mask, MaskHandler

__all__ = [
    "CvUtil"
]

from ._var import *
from ._convert import Convert
from ._mask import MaskUtil
from ._loadable_image import ImageVar

class CvUtil:
    @staticmethod
    def _apply_rotate(
        img: npt.NDArray[np.uint8], rotation: float,
        drawCallback: Callable[[np.ndarray], np.ndarray],
        center: tuple[int, int]=None, autoCenter: bool=False
    ) -> npt.NDArray[np.uint8]:
        """
        Refer to https://stackoverflow.com/a/73131334
        """
        tmp = np.zeros_like(img)
        tmp = drawCallback(tmp)
        h, w = tmp.shape[:2]
        if center is None:
            if not autoCenter:
                center = (int(w/2), int(h/2))
            else:
                a = (tmp[:,:,0] + tmp[:,:,1] + tmp[:,:,2]) > 0
                if a.sum() == 0:
                    a = (tmp[:,:,0] + tmp[:,:,1] + tmp[:,:,2]) > 0
                    assert a.sum() > 0, f"Failed to calculate alpha for autoCenter."
                y, x = np.where(a)
                cx = int(0.5 * (x.min() + x.max()))
                cy = int(0.5 * (y.min() + y.max()))
                center = (cx, cy)

        sizeRotation: tuple[int, int] = (w, h)
        affineMatrix = cv2.getRotationMatrix2D(center, rotation, 1)
        tmp = cv2.warpAffine(tmp, affineMatrix, sizeRotation, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        # Does np.uint16 help reduce error?
        tmp16 = tmp.astype(np.uint16)
        if len(tmp16.shape) == 3:
            alpha = (tmp16[:,:,0] + tmp16[:,:,1] + tmp16[:,:,2]) > 0
        elif len(tmp16.shape) == 2:
            alpha = tmp16 > 0
        else:
            raise Exception(f"Invalid shape: {tmp16.shape}")
        alpha = alpha.astype(np.float32)
        if len(tmp16.shape) == 3:
            alpha = np.dstack((alpha, alpha, alpha))
        elif len(tmp16.shape) == 2:
            pass
        else:
            raise Exception
        tmp = tmp.astype(np.float32)
        img = img.astype(np.float32)
        foreground = np.multiply(alpha, tmp)
        np.subtract(1.0, alpha, out=alpha) # invert alpha
        background = np.multiply(alpha, img)
        return np.add(foreground, background).astype(np.uint8) # There are black edges in the result. Not sure what to do about that.

    @staticmethod
    def circle(
        img: np.ndarray,
        center: VectorVar | ImageVectorCallback,
        radius: IntVar,
        color: ColorVar,
        thickness: IntVar,
        lineType: int,
        refMask: Mask=None
    ) -> np.ndarray:
        if callable(center):
            center = center(img)
        center = Convert.cast_vector(center)
        radius = Convert.cast_builtin(radius)
        color = Convert.cast_color(color)
        thickness = Convert.cast_builtin(thickness)

        if refMask is not None:
            mask = np.zeros_like(img, dtype=np.uint8)
            maskColor = (255, 255, 255)
            mask = cv2.circle(mask, center, radius, maskColor, thickness, lineType)
            mask = MaskUtil.eq_color(mask, color=maskColor)
            refMask._mask = mask
            
        return cv2.circle(img, center, radius, color, thickness, lineType)
    
    @staticmethod
    def ellipse(
        img: np.ndarray,
        center: VectorVar | ImageVectorCallback,
        axis: VectorVar,
        angle: FloatVar,
        startAngle: FloatVar,
        endAngle: FloatVar,
        color: ColorVar,
        thickness: IntVar,
        lineType: int,
        refMask: Mask=None
    ):
        if callable(center):
            center = center(img)
        center = Convert.cast_vector(center)
        axis = Convert.cast_vector(axis)
        angle = Convert.cast_builtin(angle)
        startAngle = Convert.cast_builtin(startAngle)
        endAngle = Convert.cast_builtin(endAngle)
        if startAngle > endAngle:
            tmp = startAngle; startAngle = endAngle
            endAngle = tmp
        color = Convert.cast_color(color)
        thickness = Convert.cast_builtin(thickness)

        if refMask is not None:
            mask = np.zeros_like(img, dtype=np.uint8)
            maskColor = (255, 255, 255)
            mask = cv2.ellipse(mask, center, axis, angle, startAngle, endAngle, maskColor, thickness, lineType)
            mask = MaskUtil.eq_color(mask, color=maskColor)
            refMask._mask = mask

        return cv2.ellipse(img, center, axis, angle, startAngle, endAngle, color, thickness, lineType)

    @staticmethod
    def rectangle(
        img: np.ndarray,
        pt1: VectorVar | ImageVectorCallback,
        pt2: VectorVar | ImageVectorCallback,
        color: ColorVar,
        thickness: IntVar,
        lineType: int,
        rotation: FloatVar=0,
        refMask: Mask=None
    ) -> np.ndarray:
        if callable(pt1):
            pt1 = pt1(img)
        if callable(pt2):
            pt2 = pt2(img)
        pt1 = Convert.cast_vector(pt1)
        pt2 = Convert.cast_vector(pt2)
        color = Convert.cast_color(color)
        thickness = Convert.cast_builtin(thickness)
        rotation = Convert.cast_builtin(rotation)

        center = (int(0.5 * (pt1[0] + pt2[0])), int(0.5 * (pt1[1] + pt2[1])))

        def drawCallback(img: np.ndarray, color: tuple) -> np.ndarray:
            return cv2.rectangle(img, pt1, pt2, color, thickness, lineType)

        if refMask is not None:
            mask = np.zeros_like(img, dtype=np.uint8)
            maskColor = (255, 255, 255)
            # mask = cv2.rectangle(mask, pt1, pt2, maskColor, thickness, lineType)
            if rotation != 0:
                mask = CvUtil._apply_rotate(
                    mask, rotation, partial(drawCallback, color=maskColor),
                    center=center
                )
            else:
                mask = drawCallback(mask, color=maskColor)
            mask = MaskUtil.eq_color(mask, color=maskColor)
            refMask._mask = mask

        # return cv2.rectangle(img, pt1, pt2, color, thickness, lineType)
        if rotation != 0:
            return CvUtil._apply_rotate(
                img, rotation, partial(drawCallback, color=color),
                center=center
            )
        else:
            return drawCallback(img, color=color)

    @staticmethod
    def line(
        img: np.ndarray,
        pt1: VectorVar | ImageVectorCallback,
        pt2: VectorVar | ImageVectorCallback,
        color: ColorVar,
        thickness: IntVar,
        lineType: int,
        refMask: Mask=None
    ) -> np.ndarray:
        if callable(pt1):
            pt1 = pt1(img)
        if callable(pt2):
            pt2 = pt2(img)
        pt1 = Convert.cast_vector(pt1)
        pt2 = Convert.cast_vector(pt2)
        color = Convert.cast_color(color)
        thickness = Convert.cast_builtin(thickness)

        if refMask is not None:
            mask = np.zeros_like(img, dtype=np.uint8)
            maskColor = (255, 255, 255)
            mask = cv2.line(mask, pt1, pt2, maskColor, thickness, lineType)
            mask = MaskUtil.eq_color(mask, color=maskColor)
            refMask._mask = mask

        return cv2.line(img, pt1, pt2, color, thickness, lineType)
    
    @staticmethod
    def affine_rotate(
        img: np.ndarray,
        angle: FloatVar,
        degrees: bool=True,
        scale: FloatVar=1,
        interpolation: int=cv2.INTER_LINEAR,
        adjustBorder: bool=False,
        center: VectorVar=None,
        borderColor: ColorVar=(255, 255, 255),
        maskHandler: MaskHandler=None
    ) -> np.ndarray:
        angle = Convert.cast_builtin(angle)
        scale = Convert.cast_builtin(scale)
        center = Convert.cast_vector(center)
        borderColor = Convert.cast_color(borderColor)

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

        if maskHandler is not None:
            for mask in maskHandler: # Seems to cause considerable overhead when tracking a lot of masks.
                assert mask._mask is not None
                maskBGR = MaskUtil.bool_to_bgr(mask._mask)
                maskBGR = cv2.warpAffine(maskBGR, affineMatrix, sizeRotation, flags=interpolation, borderValue=(0, 0, 0))
                mask._mask = MaskUtil.bgr_to_bool(maskBGR)

        return cv2.warpAffine(img, affineMatrix, sizeRotation, flags=interpolation, borderValue=borderColor)
    
    def text(
        img: np.ndarray, text: StringVar,
        org: VectorVar | ImageVectorCallback,
        fontFace: int=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale: FloatVar=1,
        color: ColorVar = (255, 255, 255),
        thickness: IntVar=None,
        lineType: int=None, bottomLeftOrigin: bool=False,
        rotation: FloatVar=0,
        refMask: Mask=None
    ) -> np.ndarray:
        text = Convert.cast_str(text)
        if callable(org):
            org = org(img)
        org = Convert.cast_vector(org)
        fontScale = Convert.cast_builtin(fontScale)
        color = Convert.cast_color(color)
        thickness = Convert.cast_builtin(thickness)
        rotation = Convert.cast_builtin(rotation)

        def drawCallback(img: np.ndarray, color: tuple) -> np.ndarray:
            return cv2.putText(
                img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin
            )

        if refMask is not None:
            mask = np.zeros_like(img, dtype=np.uint8)
            maskColor = (255, 255, 255)
            # mask = cv2.putText(
            #     mask, text, org, fontFace, fontScale, maskColor, thickness, lineType, bottomLeftOrigin
            # )
            if rotation != 0:
                mask = CvUtil._apply_rotate(mask, rotation, partial(drawCallback, color=maskColor), autoCenter=True)
            else:
                mask = drawCallback(mask, color=maskColor)
            mask = MaskUtil.eq_color(mask, color=maskColor)
            refMask._mask = mask

        # return cv2.putText(
        #     img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin
        # )
        if rotation != 0:
            return CvUtil._apply_rotate(img, rotation, partial(drawCallback, color=color), autoCenter=True)
        else:
            return drawCallback(img, color=color)

    @overload
    @staticmethod
    def resize(
        src: np.ndarray,
        dsize: VectorVar,
        interpolation: int=None,
        maskHandler: MaskHandler=None
    ) -> np.ndarray: ...

    @overload
    def resize(
        src: np.ndarray,
        fx: FloatVar, fy: FloatVar,
        interpolation: int=None,
        maskHandler: MaskHandler=None
    ) -> np.ndarray: ...

    @staticmethod
    def resize(
        src: np.ndarray,
        dsize: VectorVar=None,
        fx: FloatVar=None, fy: FloatVar=None,
        interpolation: int=None,
        maskHandler: MaskHandler=None
    ) -> np.ndarray:
        dsize = Convert.cast_vector(dsize)
        fx = Convert.cast_builtin(fx)
        fy = Convert.cast_builtin(fy)

        if maskHandler is not None:
            for mask in maskHandler: # Seems to cause considerable overhead when tracking a lot of masks.
                assert mask._mask is not None
                maskBGR = MaskUtil.bool_to_bgr(mask._mask)
                maskBGR = cv2.resize(maskBGR, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)
                mask._mask = MaskUtil.bgr_to_bool(maskBGR)

        return cv2.resize(src, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)

    @staticmethod
    def overlay_image(
        img: npt.NDArray[np.uint8],
        foreground: ImageVar,
        position: VectorVar | ImageVectorCallback,
        rotation: FloatVar=0,
        refMask: Mask=None, maskHandler: MaskHandler=None
    ) -> npt.NDArray[np.uint8]:
        foreground, mask = Convert.cast_image(foreground)
        if callable(position):
            position = position(img)
        position = Convert.cast_vector(position)
        rotation = Convert.cast_builtin(rotation)
        bh, bw = img.shape[:2]
        fh, fw = foreground.shape[:2]
        x, y = position

        bgBbox = BBox2D(Vector2.zero, Vector2(bw, bh))
        fgPosition = Vector2(x, y)
        fgBbox = BBox2D(fgPosition, fgPosition + Vector2(fw, fh))
        fgClampedBBox = BBox2D.Intersection(bgBbox, fgBbox)
        fgClampedBBoxZeroed = fgClampedBBox - fgClampedBBox.v0

        ry0 = fgClampedBBox.v0.y; ry1 = fgClampedBBox.v1.y
        rx0 = fgClampedBBox.v0.x; rx1 = fgClampedBBox.v1.x
        fy0 = fgClampedBBoxZeroed.v0.y; fy1 = fgClampedBBoxZeroed.v1.y
        fx0 = fgClampedBBoxZeroed.v0.x; fx1 = fgClampedBBoxZeroed.v1.x

        # result = img.copy()
        # result[ry0:ry1, rx0:rx1, :] = foreground[fy0:fy1, fx0:fx1, :]
        def drawCallback(img: np.ndarray, fg: np.ndarray, imgIdx: tuple, fgIdx: tuple) -> np.ndarray:
            result = img.copy()
            result[imgIdx] = fg[fgIdx]
            return result
        
        if rotation != 0:
            result = CvUtil._apply_rotate(
                img, rotation,
                partial(
                    drawCallback, fg=foreground,
                    imgIdx=np.index_exp[ry0:ry1, rx0:rx1, :],
                    fgIdx=np.index_exp[fy0:fy1, fx0:fx1, :]
                ),
                center=tuple(fgClampedBBox.center)
            )
        else:
            result = img.copy()
            result[ry0:ry1, rx0:rx1, :] = foreground[fy0:fy1, fx0:fx1, :]

        if rotation != 0:
            if mask is not None and refMask is not None:
                refMask._mask = np.zeros((bh, bw), dtype=np.uint8)
                refMask._mask = CvUtil._apply_rotate(
                    refMask._mask, rotation,
                    partial(
                        drawCallback,
                        fg=mask.astype(np.uint8),
                        imgIdx=np.index_exp[ry0:ry1, rx0:rx1],
                        fgIdx=np.index_exp[fy0:fy1, fx0:fx1]
                    ),
                    center=tuple(fgClampedBBox.center)
                ) > 0
            if maskHandler is not None:
                _mask = np.zeros(img.shape[:2], dtype=np.uint8)
                _mask = CvUtil._apply_rotate(
                    _mask, rotation,
                    partial(
                        drawCallback,
                        fg=np.ones(img.shape[:2], dtype=np.uint8),
                        imgIdx=np.index_exp[ry0:ry1, rx0:rx1],
                        fgIdx=np.index_exp[fy0:fy1, fx0:fx1]
                    ),
                    center=tuple(fgClampedBBox.center)
                )
                _mask = _mask > 0
                for maskObj in maskHandler:
                    assert maskObj._mask is not None
                    maskObj._mask[_mask] = False
        else:
            if mask is not None and refMask is not None:
                refMask._mask = np.zeros((bh, bw), dtype=np.bool_)
                refMask._mask[ry0:ry1, rx0:rx1] = mask[fy0:fy1, fx0:fx1]
            if maskHandler is not None:
                for maskObj in maskHandler:
                    assert maskObj._mask is not None
                    maskObj._mask[ry0:ry1, rx0:rx1] = False
        return result

    class Callback:
        @staticmethod
        def get_position_interval(img: np.ndarray) -> Interval[Vector[float]]:
            h, w = img.shape[:2]
            return Interval[Vector[float]](
                Vector[float](0, 0),
                Vector[float](w - 1, h - 1)
            )
