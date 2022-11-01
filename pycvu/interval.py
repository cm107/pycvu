from __future__ import annotations
import random
from typing import Generic, TypeVar
from .base import Base
from .vector import Vector
from .color import Color

__all__ = [
    "Interval"
]

T = TypeVar('T', int, float)
TV = TypeVar('TV', int, float, Vector, Color)

class Interval(Base, Generic[TV]):
    def __init__(self, valMin: TV, valMax: TV):
        super().__init__()
        assert type(valMin) is type(valMax)
        assert type(valMin) in TV.__constraints__, f"{type(valMin).__name__} isn't in constraints."
        if type(valMin) in T.__constraints__:
            assert valMin <= valMax
        elif type(valMin) in [Vector, Color]:
            assert type(valMin) is type(valMax)
            if type(valMin) is Vector:
                assert valMin.generic_type is valMax.generic_type
                assert valMin.x <= valMax.x and valMin.y <= valMax.y
            elif type(valMin) is Color:
                assert valMin.scale == valMax.scale
                for attr in ['r', 'g', 'b', 'a']:
                    assert getattr(valMin, attr) <= getattr(valMax, attr)
        else:
            raise TypeError

        self.valMin = valMin
        self.valMax = valMax

    @property
    def generic_type(self) -> type: # Note: Can't be called from __init__
        return self.__orig_class__.__args__[0]

    def to_dict(self) -> dict:
        item_dict = super().to_dict()
        for key in ['valMin', 'valMax']:
            assert item_dict[key]['_typedict'] == item_dict['_typedict']['_args'][0]
            del item_dict[key]['_typedict'] # Same generic type, so we can omit this redundant type info.
        return item_dict

    @classmethod
    def from_dict(cls, item_dict: dict):
        for key in ['valMin', 'valMax']:
            item_dict[key]['_typedict'] = item_dict['_typedict']['_args'][0] # Bring the omitted type info back.
        return super().from_dict(item_dict)

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
        elif self.generic_type is Color:
            r = Interval[float](self.valMin.r, self.valMax.r).random()
            g = Interval[float](self.valMin.g, self.valMax.g).random()
            b = Interval[float](self.valMin.b, self.valMax.b).random()
            a = Interval[float](self.valMin.a, self.valMax.a).random()
            return Color(r, g, b, a, self.valMin.scale)
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

        print(f"{intVecInterval.to_dict()=}")
        assert intVecInterval == Interval[Vector[int]].from_dict(intVecInterval.to_dict())
        assert intVecInterval.to_dict() == Interval[Vector[int]].from_dict(intVecInterval.to_dict()).to_dict()
        
        print("Random Color Interval")
        colorInterval = Interval[Color](Color.black, Color.white)
        for i in range(10):
            print(f"\t{i}: {colorInterval.random()=}")
        
        print(f"Test passed")
