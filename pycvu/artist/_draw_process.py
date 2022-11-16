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
from ..interval import Interval
from ..util._var import IntVar
from ..util._convert import Convert
from ..util._func import clamp
from ._img_type import ImageType

class DrawProcess(Base):
    def __init__(
        self, p: ProcessType, maskSetting: MaskSetting=None,
        weight: float=1.0, repeat: IntVar=1, prob: float=1.0,
        imgType: ImageType=ImageType.CV
    ):
        self.p = p
        self.maskSetting = maskSetting
        
        self.weight = weight
        self.repeat = repeat
        self.prob = prob
        self.imgType = imgType
    
    def to_dict(self) -> dict:
        if type(self.p) is DrawProcessGroup:
            pDict = self.p.to_dict()
        else:
            pDict = BaseUtil.to_func_dict(self.p)
        result = dict(p=pDict, weight=self.weight)
        if self.maskSetting is not None:
            result['maskSetting'] = self.maskSetting.to_dict()
        result['repeat'] = self.repeat if not hasattr(self.repeat, 'to_dict') else self.repeat.to_dict()
        result['prob'] = self.prob
        result['imgType'] = self.imgType.to_dict()
        result['_typedict'] = BaseUtil.to_type_dict(type(self))
        return result
    
    @classmethod
    def from_dict(cls, item_dict: dict, **kwargs) -> DrawProcess:
        assert 'p' in item_dict
        assert type(item_dict['p']) is dict
        if '_typedict' in item_dict['p']:
            objCls = BaseUtil.from_type_dict(item_dict['p']['_typedict'])
            assert objCls is DrawProcessGroup
            p = objCls.from_dict(item_dict['p'], **kwargs)
        else:
            p = BaseUtil.from_func_dict(item_dict['p'], **kwargs)
        if 'maskSetting' in item_dict:
            maskSetting = MaskSetting.from_dict(item_dict['maskSetting'])
        else:
            maskSetting = None
        weight = item_dict['weight']
        repeat = item_dict['repeat']
        if type(repeat) is dict:
            assert '_typedict' in repeat
            repeatCls = BaseUtil.from_type_dict(repeat['_typedict'])
            assert hasattr(repeatCls, 'from_dict')
            repeat = repeatCls.from_dict(repeat)
        prob = item_dict['prob']
        imgType = ImageType.from_dict(item_dict['imgType'])
        return DrawProcess(
            p, maskSetting,
            weight=weight,
            repeat=repeat,
            prob=prob,
            imgType=imgType
        )

    @property
    def enabled(self) -> bool:
        return random.random() <= self.prob

    @staticmethod
    def _adjust_to_img_type(img: np.ndarray | pilImage.Image, imgType: ImageType) -> np.ndarray | pilImage.Image:
        currentImgType = ImageType._value2member_map_[type(img)]
        if imgType != currentImgType:
            if currentImgType == ImageType.PIL and imgType == ImageType.CV:
                return Convert.pil_to_cv(img)
            elif currentImgType == ImageType.CV and imgType == ImageType.PIL:
                return Convert.cv_to_pil(img)
            else:
                raise ValueError
        else:
            return img.copy()

    def draw(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # out = img.copy()
        out = DrawProcess._adjust_to_img_type(img, self.imgType)
        hook = self.p
        currentImgType = ImageType._value2member_map_[type(out)]
        if self.imgType != currentImgType:
            if currentImgType == ImageType.PIL and self.imgType == ImageType.CV:
                out = Convert.pil_to_cv(out)
            elif currentImgType == ImageType.CV and self.imgType == ImageType.PIL:
                out = Convert.cv_to_pil(out)
            else:
                raise ValueError

        def _draw(out: np.ndarray) -> np.ndarray:
            if self.enabled:
                repeat = self.repeat if type(self.repeat) is not Interval else self.repeat.random()
                for i in range(repeat):
                    if type(hook) is partial:
                        out = hook(out)
                    elif type(hook) is DrawProcessGroup:
                        gHook = hook._preproc.preprocess(hook)
                        for h in gHook:
                            h: DrawProcess = h # Type hints not working here?
                            out = h.draw(out)
                    elif callable(hook):
                        out = hook(out)
                    else:
                        raise TypeError
            return out
        
        return _draw(out)
    
    def draw_and_get_mask(
        self, img: npt.NDArray[np.uint8],
        maskHandler: MaskHandler
    ) -> npt.NDArray[np.uint8]:
        # out = img.copy()
        out = DrawProcess._adjust_to_img_type(img, self.imgType)
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
            if self.enabled:
                repeat = self.repeat if type(self.repeat) is not Interval else self.repeat.random()
                for i in range(repeat):
                    if type(hook) is partial:
                        fargs: list[str] = inspect.getfullargspec(hook.func).args
                        kwargs = _get_kwargs(fargs, hook.func.__qualname__)
                        out = hook(out, **kwargs)
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

        # Does not seems to be helping.
        # def _draw_parallel(out: np.ndarray) -> np.ndarray:
        #     from joblib import Parallel, delayed
        #     if self.enabled:
        #         repeat = self.repeat if type(self.repeat) is not Interval else self.repeat.random()
        #         if type(hook) is partial:
        #             fargs: list[str] = inspect.getfullargspec(hook.func).args
        #             if repeat == 1 or 'maskHandler' in fargs or type(out) is not np.ndarray:
        #                 for i in range(repeat):
        #                     kwargs = _get_kwargs(fargs, hook.func.__qualname__)
        #                     out = hook(out, **kwargs)
        #                     if 'refMask' in kwargs:
        #                         maskHandler.process(kwargs['refMask'])
        #             else:
        #                 def _inner(out: np.ndarray, hook: ProcessType, maskSetting: MaskSetting) -> tuple[np.ndarray, Mask]:
        #                     refMask = Mask(setting=maskSetting)
        #                     out = hook(out.copy(), refMask=refMask)
        #                     return out, refMask

        #                 results = Parallel(n_jobs=2)(delayed(_inner)(np.zeros_like(out), hook, self.maskSetting.copy()) for i in range(repeat))
        #                 for tmpImg, refMask in results:
        #                     out[refMask._mask] = tmpImg[refMask._mask]
        #                     if 'refMask' in fargs:
        #                         maskHandler.process(refMask)
        #         elif type(hook) is DrawProcessGroup:
        #             gHook = hook._preproc.preprocess(hook)
        #             for i in range(repeat):
        #                 for h in gHook:
        #                     h: DrawProcess = h # Type hints not working here?
        #                     out = h.draw_and_get_mask(out, maskHandler)
        #         elif callable(hook):
        #             fargs: list[str] = inspect.getfullargspec(hook).args
        #             for i in range(repeat):
        #                 kwargs = _get_kwargs(fargs, hook.__qualname__)
        #                 out = hook(out, **kwargs)
        #                 if 'refMask' in kwargs:
        #                     maskHandler.process(kwargs['refMask'])
        #         else:
        #             raise TypeError
        #     return out
        
        # return _draw_parallel(out)
        return _draw(out)

class DrawProcessGroup(BaseHandler[DrawProcess]):
    def __init__(
        self, _objects: list[DrawProcess]=None,
        _preproc: DrawPreprocessQueue=None
    ):
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
            out = dp.draw(out)
        out = DrawProcess._adjust_to_img_type(out, ImageType.CV)
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
        out = DrawProcess._adjust_to_img_type(out, ImageType.CV)
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
    | DrawProcessGroup