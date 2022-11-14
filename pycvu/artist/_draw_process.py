from __future__ import annotations
from typing import Callable
from functools import partial
import inspect
import numpy as np
import numpy.typing as npt
from PIL import Image as pilImage
import random

from ..base import Base, BaseHandler, BaseUtil
from ..mask import MaskSetting, Mask, MaskHandler
from ..util import RepeatDrawCallback
from ..util._var import IntVar
from ..util._convert import Convert
from ..util._func import clamp

class DrawProcess(Base):
    def __init__(self, p: ProcessType, maskSetting: MaskSetting=None, weight: float=1.0):
        self.p = p
        self.maskSetting = maskSetting
        self.weight = weight
    
    def to_dict(self) -> dict:
        if type(self.p) in [RepeatDrawCallback, DrawProcessGroup]:
            pDict = self.p.to_dict()
        else:
            pDict = BaseUtil.to_func_dict(self.p)
        result = dict(p=pDict, weight=self.weight)
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
        weight = item_dict['weight']
        return DrawProcess(p, maskSetting, weight)

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
                gHook = hook._preproc.preprocess(hook)
                for h in gHook:
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
                gHook = hook._preproc.preprocess(hook)
                for h in gHook:
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
    def __init__(self, _objects: list[DrawProcess]=None, _preproc: DrawPreprocessQueue=None):
        super().__init__(_objects)
        self._preproc: DrawPreprocessQueue = _preproc if _preproc is not None else DrawPreprocessQueue()

    @property
    def totalWeight(self) -> float:
        return sum([dp.weight for dp in self])

class DrawProcessQueue(BaseHandler[DrawProcess]):
    def __init__(self, _objects: list[DrawProcess]=None):
        super().__init__(_objects)

    @property
    def totalWeight(self) -> float:
        return sum([dp.weight for dp in self])

    def draw(
        self, img: npt.NDArray[np.uint8],
        preprocess: ProcQueueMap=None
    ) -> npt.NDArray[np.uint8] | None:
        out = img.copy()

        if preprocess is not None:
            q = preprocess(self)
        else:
            q = self
        for dp in q:
            dp.draw(out, out=out)
        return out

    def draw_and_get_masks(
        self, img: npt.NDArray[np.uint8],
        maskHandler: MaskHandler=None,
        preprocess: ProcQueueMap=None
    ) -> tuple[npt.NDArray[np.uint8] | None, MaskHandler | None]:
        out = img.copy()
        if maskHandler is None:
            maskHandler = MaskHandler()
        
        if preprocess is not None:
            q = preprocess(self)
        else:
            q = self
        for dp in q:
            out = dp.draw_and_get_mask(
                img=out,
                maskHandler=maskHandler
            )
        
        return (out, maskHandler)

class DrawPreprocessQueue(Base):
    def __init__(self):
        self._opQueue: list[ProcQueueMap] = []

    def to_dict(self) -> dict:
        return dict(
            _opQueue=[BaseUtil.to_func_dict(p) for p in self._opQueue],
            _typedict=BaseUtil.to_type_dict(type(self))
        )

    @classmethod
    def from_dict(cls, item_dict: dict, **kwargs) -> DrawPreprocessQueue:
        obj = DrawPreprocessQueue()
        obj._opQueue = [BaseUtil.from_func_dict(val, **kwargs) for val in item_dict['_opQueue']]
        return obj

    @staticmethod
    def _shuffle(queue: Q) -> Q:
        result = queue.copy()
        result.shuffle()
        return result

    def shuffle(self):
        p = partial(self._shuffle)
        self._opQueue.append(p)
    
    @staticmethod
    def _some_of(queue: Q, n: IntVar) -> Q:
        totalWeight = queue.totalWeight
        weights = [dp.weight / totalWeight for dp in queue]
        n = Convert.cast_builtin(n)
        n = clamp(n, 0, len(queue))
        return type(queue)(random.choices(queue, k=n, weights=weights))

    def some_of(self, n: IntVar):
        p = partial(self._some_of, n=n)
        self._opQueue.append(p)

    @staticmethod
    def _one_of(queue: DrawProcessQueue) -> DrawProcessQueue:
        return DrawPreprocessQueue._some_of(queue, n=1)

    def one_of(self):
        p = partial(self._one_of)
        self._opQueue.append(p)

    @property
    def preprocess(self) -> ProcQueueMap:
        def _apply(queue: Q) -> Q:
            q = queue.copy()
            for p in self._opQueue:
                q = p(q)
            return q
        return _apply

from typing import TypeVar

DrawQueueType = DrawProcessQueue | DrawProcessGroup
Q = TypeVar('Q', bound=DrawQueueType)
ProcQueueMap = Callable[[Q], Q]
ProcInOutType = npt.NDArray[np.uint8] | pilImage.Image
ProcessType = Callable[[ProcInOutType], ProcInOutType] \
    | RepeatDrawCallback | DrawProcessGroup