from __future__ import annotations
from typing import Callable
import cv2
import numpy as np
import numpy.typing as npt
from pyevu import BBox2D, Vector2
from ..interval import Interval
from ..color import Color, HSV

__all__ = ["MaskUtil"]

class MaskUtil:
    @staticmethod
    def occlude(subMask: npt.NDArray[np.bool_], fromMask: npt.NDArray[np.bool_]):
        fromMask[subMask & fromMask] = False

    @staticmethod
    def eq_color(img: np.ndarray, color: tuple[int, int, int] | Color) -> npt.NDArray[np.bool_]:
        if type(color) is Color:
            color = color.bgr
        return np.all(img == color, axis=2)

    @staticmethod
    def neq_color(img: np.ndarray, color: tuple[int, int, int] | Color) -> npt.NDArray[np.bool_]:
        if type(color) is Color:
            color = color.bgr
        return ~MaskUtil.eq_color(img, color)

    @staticmethod
    def bool_to_bgr(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
        maskImg = np.zeros(tuple(list(mask.shape) + [3]))
        maskImg[mask] = (255, 255, 255)
        return maskImg
    
    @staticmethod
    def bgr_to_bool(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        return MaskUtil.neq_color(mask, (0, 0, 0))

    @staticmethod
    def from_img(img: npt.NDArray[np.uint8], color: tuple[int, int, int] | Color | Interval[Color] | Interval[HSV]) -> npt.NDArray[np.bool_]:
        if type(color) is not Interval:
            return MaskUtil.eq_color(img=img, color=color)
        else:
            if color.generic_type is Color:
                def contains(value: npt.NDArray[np.uint8]) -> bool:
                    return np.array([color.contains(Color(*value.tolist()))])
            elif color.generic_type is HSV:
                def contains(value: npt.NDArray[np.uint8]) -> bool:
                    return np.array([color.contains(Color(*value.tolist()).to_hsv())])
            else:
                raise TypeError
            np_contains: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.bool_]] = np.vectorize(contains, signature='(xyz)->(xy)')
            return np_contains(img).reshape(img.shape[:2])

    @staticmethod
    def bbox_from_mask(mask: npt.NDArray[np.bool_]) -> BBox2D:
        if mask is None:
            return None
        y, x = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            return None
        return BBox2D(Vector2(x.min().tolist(), y.min().tolist()), Vector2(x.max().tolist(), y.max().tolist()))

    @staticmethod
    def debug():
        symbol_dir = "/home/clayton/workspace/git/pycvu/test/symbol"
        path = f"{symbol_dir}/A1_21_label0_1.png"
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # mask = MaskUtil.from_img(img, color=Interval[Color](Color(0,0,0), Color(200,200,200)))
        mask = MaskUtil.from_img(img, color=Interval[HSV](HSV(0,0,0), HSV(359.9, 1, 0.9)))
        maskImg = np.zeros(tuple(list(mask.shape) + [3]))
        maskImg[mask] = (255, 255, 255)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', int(img.shape[1]*4), int(img.shape[0]*4))
        cv2.imshow('img', img)

        cv2.namedWindow('maskImg', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('maskImg', int(maskImg.shape[1]*4), int(maskImg.shape[0]*4))
        cv2.imshow('maskImg', maskImg)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
