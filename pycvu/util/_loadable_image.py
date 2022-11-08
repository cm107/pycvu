from __future__ import annotations
import numpy as np
import numpy.typing as npt
import cv2
import glob
import random
from ..base import Base, BaseHandler, ContextVarRef
from ..interval import Interval
from ..color import Color, HSV
from ._mask import MaskUtil

class LoadableImage(Base):
    def __init__(self, path: str):
        self.path = path
        self._img: npt.NDArray[np.uint8] = None

    def load_data(self):
        self._img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)[:,:,:3]
        # self._img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        # self._img = cv2.imread(self.path)
    
    def clear_data(self):
        self._img = None

    @property
    def loaded(self) -> bool:
        return self._img is not None

    @property
    def data(self) -> npt.NDArray[np.uint8]:
        assert self._img is not None, "Data hasn't been loaded yet."
        return self._img

class LoadableImageHandler(BaseHandler[LoadableImage]):
    def __init__(self, _objects: list[LoadableImageMask]=None):
        super().__init__(_objects)
    
    def load_data(self):
        for obj in self:
            obj.load_data()
    
    def clear_data(self):
        for obj in self:
            obj.clear_data()

    @classmethod
    def from_wildcard(cls, wildcardPath: str) -> LoadableImageHandler:
        paths = glob.glob(wildcardPath)
        return LoadableImageHandler([LoadableImage(path) for path in paths])

    def random(self) -> LoadableImage:
        return random.choice(self)

class LoadableImageMask(LoadableImage):
    def __init__(self, path: str, maskThresh: Interval[Color] | Interval[HSV]):
        super().__init__(path=path)
        assert maskThresh.generic_type in [Color, HSV]
        self.maskThresh = maskThresh
        self._mask: npt.NDArray[np.bool_] = None

    def to_dict(self, includeData: bool=True) -> dict:
        item_dict = super().to_dict()
        if not includeData:
            del item_dict['_img']
            del item_dict['_mask']
        else:
            assert type(item_dict['_img']) is np.ndarray and item_dict['_img'].dtype is np.dtype('uint8'), f"{item_dict['_img'].dtype=}"
            item_dict['_img'] = item_dict['_img'].tolist()
            assert type(item_dict['_mask']) is np.ndarray and item_dict['_mask'].dtype is np.dtype('bool_'), f"{item_dict['_mask'].dtype=}"
            item_dict['_mask'] = item_dict['_mask'].tolist()

        return item_dict

    def save(self, path: str, includeData: bool=True):
        super().save(path=path, includeData=includeData)

    @classmethod
    def from_dict(cls, item_dict: dict) -> LoadableImageMask:
        if '_img' in item_dict:
            item_dict['_img'] = np.array(item_dict['_img']).astype(np.uint8)
        if '_mask' in item_dict:
            item_dict['_mask'] = np.array(item_dict['_mask']).astype(np.bool_)
        return super().from_dict(item_dict)

    def load_data(self):
        super().load_data()
        self._mask = MaskUtil.from_img(self._img, self.maskThresh)
    
    def clear_data(self):
        super().clear_data()
        self._mask = None
    
    @property
    def loaded(self) -> bool:
        return self._img is not None and self._mask is not None

    @property
    def data(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
        return self._img, self._mask

    def show_preview(self, zoom: float=1.0):
        img, mask = self.data
        if True:
            maskImg = np.zeros_like(img)
            maskImg[mask] = (255, 255, 255)
            preview = np.concatenate([img, maskImg], axis=0, dtype=np.uint8)
        else:
            bg = np.zeros((img.shape[0]*2, img.shape[1]*2, img.shape[2])).astype(np.uint8)
            bg[:,:,:] = (0,0,255)
            from ..util import ImageUtil
            bg = ImageUtil.overlay_image(img, bg, position=(0,0))
            preview = bg
        cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('preview', int(preview.shape[1]*zoom), int(preview.shape[0]*zoom))
        cv2.imshow('preview', preview)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    @staticmethod
    def debug():
        symbol_dir = "/home/clayton/workspace/git/pycvu/test/symbol"
        obj = LoadableImageMask(
            f"{symbol_dir}/A1_21_label0_1.png",
            Interval[HSV](HSV(0,0,0), HSV(359.9, 1, 0.9))
        )
        print(f"{obj.to_dict()=}")
        obj.load_data()
        print(f"{obj.to_dict()=}")
        assert obj.to_dict() == LoadableImageMask.from_dict(obj.to_dict()).to_dict()
        obj.show_preview(zoom=4)

class LoadableImageMaskHandler(BaseHandler[LoadableImageMask]):
    def __init__(self, _objects: list[LoadableImageMask]=None):
        super().__init__(_objects)
    
    def to_dict(self, includeData: bool=True) -> dict:
        return super().to_dict(includeData=includeData)
    
    def save(self, path: str, includeData: bool=True):
        super().save(path=path, includeData=includeData)

    def load_data(self):
        for obj in self:
            obj.load_data()
    
    def clear_data(self):
        for obj in self:
            obj.clear_data()

    @classmethod
    def from_wildcard(cls, wildcardPath: str, maskThresh: Interval[Color] | Interval[HSV]) -> LoadableImageMaskHandler:
        paths = glob.glob(wildcardPath)
        return LoadableImageMaskHandler([LoadableImageMask(path, maskThresh) for path in paths])

    def random(self) -> LoadableImageMask:
        return random.choice(self)

    @staticmethod
    def debug():
        handler = LoadableImageMaskHandler.from_wildcard(
            "/home/clayton/workspace/git/pycvu/test/symbol/*.png",
            Interval[HSV](HSV(0,0,0), HSV(359.9, 1, 0.9))
        )
        handler.load_data()
        # for obj in handler:
        #     print(f"{obj._img.shape=}, {obj._mask.shape=}")
        #     obj.show_preview(zoom=4)
        handler.random().show_preview(zoom=4)

ImageVar = npt.NDArray[np.uint8] | LoadableImage | LoadableImageMask | LoadableImageHandler | LoadableImageMaskHandler | ContextVarRef
