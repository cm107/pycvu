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

class DrawProcess(Base):
    def __init__(self, p: ProcessType, maskSetting: MaskSetting=None):
        self.p = p
        self.maskSetting = maskSetting
    
    def to_dict(self) -> dict:
        if type(self.p) in [RepeatDrawCallback, DrawProcessGroup]:
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
            objCls = BaseUtil.from_type_dict(item_dict['p']['_typedict'])
            assert objCls in [RepeatDrawCallback, DrawProcessGroup]            
            p = objCls.from_dict(item_dict['p'], **kwargs)
        else:
            p = BaseUtil.from_func_dict(item_dict['p'], **kwargs)
        if 'maskSetting' in item_dict:
            maskSetting = MaskSetting.from_dict(item_dict['maskSetting'])
        else:
            maskSetting = None
        return DrawProcess(p, maskSetting)

    def draw(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        out = img.copy()
        hook = self.p

        def _draw(out: np.ndarray) -> np.ndarray:
            if type(hook) is partial:
                out = hook(out)
            elif type(hook) is RepeatDrawCallback:
                for i in range(hook.repeat):
                    out = hook.p(out)
            elif type(hook) is DrawProcessGroup:
                for h in hook:
                    h: DrawProcess = h # Type hints not working here?
                    out = h.draw(out)
            elif callable(hook):
                out = hook(out)
            else:
                raise TypeError
        
        return _draw(out)
    
    def draw_and_get_mask(
        self, img: npt.NDArray[np.uint8],
        maskHandler: MaskHandler
    ) -> npt.NDArray[np.uint8]:
        out = img.copy()
        hook = self.p

        def _get_kwargs(fargs: list[str], qualname: str) -> dict:
            kwargs = dict()
            if self.maskSetting is not None:
                if 'refMask' in fargs:
                    kwargs['refMask'] = Mask(setting=self.maskSetting)
                else:
                    print(f"TODO: Implement refMask for {qualname}")
            if 'maskHandler' in fargs:
                kwargs['maskHandler'] = maskHandler
            return kwargs

        def _draw(out: np.ndarray) -> np.ndarray:
            if type(hook) is partial:
                fargs: list[str] = inspect.getfullargspec(hook.func).args
                kwargs = _get_kwargs(fargs, hook.func.__qualname__)
                out = hook(out, **kwargs)
                if 'refMask' in kwargs:
                    maskHandler.process(kwargs['refMask'])
            elif type(hook) is RepeatDrawCallback:
                fargs: list[str] = inspect.getfullargspec(hook.p.func).args
                for i in range(hook.repeat):
                    kwargs = _get_kwargs(fargs, hook.p.func.__qualname__)
                    out = hook.p(out, **kwargs)
                    if 'refMask' in kwargs:
                        maskHandler.process(kwargs['refMask'])
            elif type(hook) is DrawProcessGroup:
                for h in hook:
                    h: DrawProcess = h # Type hints not working here?
                    out = h.draw_and_get_mask(out, maskHandler)
            elif callable(hook):
                fargs: list[str] = inspect.getfullargspec(hook).args
                kwargs = _get_kwargs(fargs, hook.__qualname__)
                out = hook(out, **kwargs)
                if 'refMask' in kwargs:
                    maskHandler.process(kwargs['refMask'])
            else:
                raise TypeError
            return out
        
        return _draw(out)

class DrawProcessGroup(BaseHandler[DrawProcess]):
    def __init__(self, _objects: list[DrawProcess]=None):
        super().__init__(_objects)

    def to_dict(self, compressed: bool=True, **kwargs):
        # print(f"\n{type(self).__name__}.to_dict")
        result = super().to_dict(compressed=compressed, **kwargs)
        # print(f"{type(self).__name__} {result=}")
        return result

    @classmethod
    def from_dict(cls, item_dict: dict, compressed: bool=True, **kwargs) -> DrawProcessGroup:
        # print(f"\n{cls.__name__}.from_dict")
        # print(f"{cls.__name__} {item_dict=}")
        return super().from_dict(item_dict, compressed=compressed, **kwargs)

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

ProcInOutType = npt.NDArray[np.uint8] | pilImage.Image
ProcessType = Callable[[ProcInOutType], ProcInOutType] \
    | RepeatDrawCallback | DrawProcessGroup