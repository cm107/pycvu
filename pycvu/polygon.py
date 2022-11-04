from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Generator, cast, Callable

from .base import Base, BaseHandler
from .vector import Vector

class Polygon(Base):
    def __init__(self, data: list[int]):
        assert len(data) % 2 == 0
        self.data = data
    
    def __len__(self) -> int:
        return int(len(self.data) // 2)
    
    def __iter__(self) -> Generator[Vector[int]]:
        for i in range(len(self)):
            yield self[i]

    def _reformat_slice(self, idx: slice) -> list[int]:
        start = idx.start if idx.start is not None else 0
        if start < 0:
            start = len(self) + start % len(self)
        stop = idx.stop if idx.stop is not None else len(self)
        if stop < 0:
            stop = len(self) + stop % len(self)
        step = idx.step if idx.step is not None else 1
        assert step != 0
        if step > 0:
            ilist = list(range(start, stop, step))
        else:
            ilist = list(range(start, stop, -step))[::-1]
        return ilist

    def __getitem__(self, idx):
        if type(idx) is int:
            return Vector(self.data[idx*2], self.data[idx*2+1])
        elif type(idx) is slice:
            result: list[Vector[int]] = []
            for i in self._reformat_slice(idx):
                result.append(Vector(self.data[i*2], self.data[i*2+1]))
            return result
        else:
            raise TypeError

    def __setitem__(self, idx, value):
        if type(idx) is int:
            assert type(value) is Vector
            self.data[idx*2] = value.x
            self.data[idx*2+1] = value.y
        elif type(idx) is slice:
            assert type(value) is list
            ilist = self._reformat_slice(idx)
            assert len(ilist) == len(value)
            value = cast(list[Vector[int]], value)
            for i, j in enumerate(ilist):
                self.data[j*2] = value[i].x
                self.data[j*2+1] = value[i].y
        else:
            raise TypeError

    def __delitem__(self, idx):
        if type(idx) is int:
            del self.data[idx*2:idx*2+2:1]
        elif type(idx) is slice:
            ilist = self._reformat_slice(idx)
            if ilist[0] < ilist[-1]:
                ilist = ilist[::-1]
            for i in ilist:
                del self.data[i*2:i*2+2:1]
        else:
            raise TypeError

    def to_contour(self) -> npt.NDArray[np.int32]:
        return np.array(self.data).reshape((-1, 1, 2)).astype(np.int32)

    @classmethod
    def from_contour(cls, contour: npt.NDArray[np.int32]) -> Polygon:
        return Polygon(contour.reshape(-1).tolist())

    def append(self, obj):
        if type(obj) is Vector:
            self.data.extend(list(obj))
        elif type(obj) in [tuple, list]:
            assert len(obj) % 2 == 0
            self.data.extend(list(obj))
        else:
            raise TypeError

    @staticmethod
    def debug():
        poly = Polygon([0,1,2,3,4,5,6,7,8,9])
        del poly[2]
        poly[0] += 10
        poly.append(Vector(100, 100))
        poly.append((200, 200))
        print(f"{len(poly)=}")
        for p in poly[::-1]:
            print(f"{p=}")

class Segmentation(BaseHandler[Polygon]):
    def __init__(self, _objects: list[Polygon]=None):
        super().__init__(_objects)

    def to_contours(self) -> tuple:
        return tuple([poly.to_contour() for poly in self])

    @classmethod
    def from_contours(cls, contours: tuple) -> Segmentation:
        return Segmentation([Polygon.from_contour(contour) for contour in contours])

    def to_coco(self) -> list[list[int]]:
        return [poly.data for poly in self]
    
    @classmethod
    def from_coco(cls, data: list[list[int]]) -> Segmentation:
        return Segmentation([Polygon(polyData) for polyData in data])

    def prune(self, condition: Callable[[Polygon], bool]) -> Segmentation:
        seg = self.deepcopy()
        for i in list(range(len(seg)))[::-1]:
            if condition(seg[i]):
                del seg[i]
        return seg
