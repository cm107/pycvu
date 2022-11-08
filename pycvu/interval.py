from __future__ import annotations
import random
from typing import Generic, TypeVar
from .base import Base
from .vector import Vector
from .color import Color, HSV

__all__ = [
    "Interval"
]

T = TypeVar('T', int, float)
TV = TypeVar('TV', int, float, Vector, Color, HSV)

class Interval(Base, Generic[TV]):
    def __init__(self, valMin: TV, valMax: TV):
        super().__init__()
        if type(valMin) is float and type(valMax) is int:
            valMax = float(valMax)
        elif type(valMin) is int and type(valMax) is float:
            valMin = float(valMin)
        assert type(valMin) is type(valMax), f"{type(valMin)=} is not {type(valMax)=}"
        assert type(valMin) in TV.__constraints__, f"{type(valMin).__name__} isn't in constraints."
        if type(valMin) in T.__constraints__:
            assert valMin <= valMax
        elif type(valMin) in [Vector, Color, HSV]:
            assert type(valMin) is type(valMax)
            if type(valMin) is Vector:
                assert valMin.generic_type is valMax.generic_type
                assert valMin.x <= valMax.x and valMin.y <= valMax.y
            elif type(valMin) is Color:
                assert valMin.scale == valMax.scale
                for attr in ['r', 'g', 'b', 'a']:
                    assert getattr(valMin, attr) <= getattr(valMax, attr)
            elif type(valMin) is HSV:
                for attr in ['h', 's', 'v']:
                    assert getattr(valMin, attr) <= getattr(valMax, attr)
            else:
                raise Exception
        else:
            raise TypeError

        self.valMin = valMin
        self.valMax = valMax

    def contains(self, value: TV) -> bool:
        assert type(self.valMin) is type(value)
        if type(value) in T.__constraints__:
            return self.valMin <= value <= self.valMax
        elif type(value) in TV.__constraints__:
            def _contains(value: TV, attrList: list[str]) -> bool:
                return all([getattr(self.valMin, attr) <= getattr(value, attr) <= getattr(self.valMax, attr) for attr in attrList])
            
            if type(value) is Vector:
                return _contains(value, ['x', 'y'])
            elif type(value) is Color:
                assert self.valMin.scale == value.scale
                return _contains(value, ['r', 'g', 'b', 'a'])
            elif type(value) is HSV:
                return _contains(value, ['h', 's', 'v'])
            else:
                raise Exception
        else:
            raise TypeError

    @property
    def generic_type(self) -> type: # Note: Can't be called from __init__
        return self.__orig_class__.__args__[0]

    def to_dict(self) -> dict:
        item_dict = super().to_dict()
        _boundTypeOmitted = False
        for key in ['valMin', 'valMax']:
            if type(item_dict[key]) is dict and '_typedict' in item_dict[key]:
                assert item_dict[key]['_typedict'] == item_dict['_typedict']['_args'][0]
                del item_dict[key]['_typedict'] # Same generic type, so we can omit this redundant type info.
                _boundTypeOmitted = True
        item_dict['_boundTypeOmitted'] = _boundTypeOmitted
        return item_dict

    @classmethod
    def from_dict(cls, item_dict: dict):
        if item_dict['_boundTypeOmitted']:
            for key in ['valMin', 'valMax']:
                item_dict[key]['_typedict'] = item_dict['_typedict']['_args'][0] # Bring the omitted type info back.
        del item_dict['_boundTypeOmitted']
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
        elif self.generic_type is HSV:
            h = Interval[float](self.valMin.h, self.valMax.h).random()
            s = Interval[float](self.valMin.s, self.valMax.s).random()
            v = Interval[float](self.valMin.v, self.valMax.v).random()
            return HSV(h, s, v)
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
        
        assert Interval[Vector[float]](Vector[float](0,0), Vector[float](10,10)).contains(Vector[float](5,5))
        assert not Interval[Vector[float]](Vector[float](0,0), Vector[float](10,10)).contains(Vector[float](15,15))
        assert Interval[Color](Color(0,0,0), Color(100, 100, 100)).contains(Color(50, 50, 50))
        assert not Interval[Color](Color(0,0,0), Color(100, 100, 100)).contains(Color(150, 150, 150))
        assert Interval[HSV](HSV(0, 0, 0), HSV(100, 0.5, 0.5)).contains(HSV(50, 0.1, 0.1))
        assert not Interval[HSV](HSV(0, 0, 0), HSV(100, 0.5, 0.5)).contains(HSV(50, 0.6, 0.1))

        print(f"Test passed")
