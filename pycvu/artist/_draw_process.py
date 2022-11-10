from __future__ import annotations
from typing import Callable
from functools import partial
import inspect
import numpy as np
import numpy.typing as npt
from PIL import Image as pilImage

from ..base import Base, BaseHandler, BaseUtil
from ..util import RepeatDrawCallback
from ..mask import MaskSetting, Mask, MaskHandler

ProcInOutType = npt.NDArray[np.uint8] | pilImage.Image
ProcessType = Callable[[ProcInOutType], ProcInOutType] | RepeatDrawCallback

class DrawProcess(Base):
    def __init__(self, p: ProcessType, maskSetting: MaskSetting=None):
        self.p = p
        self.maskSetting = maskSetting
    
    def to_dict(self) -> dict:
        if type(self.p) is RepeatDrawCallback:
            pDict = self.p.to_dict()
        else:
            pDict = BaseUtil.to_func_dict(self.p)
        result = dict(p=pDict)
        if self.maskSetting is not None:
            result['maskSetting'] = self.maskSetting.to_dict()
        result['_typedict'] = BaseUtil.to_type_dict(type(self))
        return result
    
    @classmethod
    def from_dict(cls, item_dict: dict, **kwargs) -> DrawProcess:
        assert 'p' in item_dict
        assert type(item_dict['p']) is dict
        if '_typedict' in item_dict['p']:
            assert item_dict['p']['_typedict']['_qualname'] == RepeatDrawCallback.__name__
            p = RepeatDrawCallback.from_dict(item_dict['p'], **kwargs)
        else:
            p = BaseUtil.from_func_dict(item_dict['p'], **kwargs)
        if 'maskSetting' in item_dict:
            maskSetting = MaskSetting.from_dict(item_dict['maskSetting'])
        else:
            maskSetting = None
        return DrawProcess(p, maskSetting)

    def draw(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        out = img.copy()
        if type(self.p) is partial:
            hook = RepeatDrawCallback(self.p)
        else:
            hook = self.p
        
        for i in range(hook.repeat):
            out = hook.p(out)
        return out
    
    def draw_and_get_mask(
        self, img: npt.NDArray[np.uint8],
        maskHandler: MaskHandler
    ) -> npt.NDArray[np.uint8]:
        out = img.copy()
        if type(self.p) is partial:
            hook = RepeatDrawCallback(self.p)
        else:
            hook = self.p
        
        funcArgs: list[str] = inspect.getfullargspec(hook.p.func).args
        if (
            self.maskSetting is not None
            and 'refMask' in funcArgs
            and 'maskHandler' in funcArgs
        ):
            for i in range(hook.repeat):
                mask = Mask(setting=self.maskSetting)
                out = hook.p(out, refMask=mask, maskHandler=maskHandler)
                maskHandler.process(mask)
        elif self.maskSetting is not None and 'refMask' in funcArgs:
            for i in range(hook.repeat):
                mask = Mask(setting=self.maskSetting)
                out = hook.p(out, refMask=mask)
                maskHandler.process(mask)
        elif 'maskHandler' in funcArgs:
            for i in range(hook.repeat):
                out = hook.p(out, maskHandler=maskHandler)
        elif self.maskSetting is not None:
            print(f"TODO: Implement refMask for {hook.p.func.__qualname__}")
            for i in range(hook.repeat):
                out = hook.p(out)
        else:
            for i in range(hook.repeat):
                out = hook.p(out)
        
        return out

class DrawProcessQueue(BaseHandler[DrawProcess]):
    def __init__(self, _objects: list[DrawProcess]=None):
        super().__init__(_objects)

    def draw(
        self, img: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8] | None:
        out = img.copy()
        for dp in self:
            dp.draw(out, out=out)
        return out

    def draw_and_get_masks(
        self, img: npt.NDArray[np.uint8],
        maskHandler: MaskHandler=None
    ) -> tuple[npt.NDArray[np.uint8] | None, MaskHandler | None]:
        out = img.copy()
        if maskHandler is None:
            maskHandler = MaskHandler()
        
        for dp in self:
            out = dp.draw_and_get_mask(
                img=out,
                maskHandler=maskHandler
            )
        
        return (out, maskHandler)
