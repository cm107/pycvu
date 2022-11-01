from __future__ import annotations
from typing import Generic, TypeVar
from pyevu import Vector2
from .base import Base, BaseHandler

__all__ = [
    "Vector"
]

T = TypeVar('T', int, float)

class Vector(Vector2, Base, Generic[T]):
    def __init__(self, x: T, y: T):
        assert type(x) is type(y)
        assert type(x) in T.__constraints__
        Vector2.__init__(self, x=x, y=y)
        Base.__init__(self)
    
    def __str__(self) -> str:
        cls_str = f"{type(self).__name__}[{self.generic_type.__name__}]" if hasattr(self, '__orig_class__') else f"{type(self).__name__}"
        return f"{cls_str}({','.join([f'{key}={val}' for key, val in self.__dict__.items() if key != '__orig_class__'])})"

    @property
    def generic_type(self) -> type:
        # Note: Can't be called from __init__ or any other dunder method.
        # Also cannot be called from any classmethod.
        # When instance is created with cls() in a classmethod, __orig_class__ doesn't seem to be initialized properly.
        # Avoid using Vector[int].zero, Vector[int].one, and so on when the generic type needs to be checked.
        return self.__orig_class__.__args__[0]

    @staticmethod
    def debug():
        intVec = Vector[int](1, 2)
        print(f"{intVec=}")
        print(f"{intVec.to_dict()=}")
        print(f"{Vector[int].from_dict(intVec.to_dict())=}")
        intVecCopy = Vector[int].from_dict(intVec.to_dict())

class IntVectorList(BaseHandler[Vector[int]]): # For debugging
    def __init__(self, _objects: list[Vector[int]]=None):
        super().__init__(_objects)
    
    @staticmethod
    def debug():
        vectors = IntVectorList([
            Vector[int](1, 2), Vector[int](3, 4),
            Vector[int](5, 6), Vector[int](7, 8),
        ])
                
        print(f"{vectors.to_dict()=}")
        print(f"{IntVectorList.from_dict(vectors.to_dict())=}")
        assert vectors.to_dict() == IntVectorList.from_dict(vectors.to_dict()).to_dict()
        assert vectors == IntVectorList.from_dict(vectors.to_dict())
        print('Test passed')

class GenericVectorList(BaseHandler[Vector[T]]): # For debugging
    def __init__(self, _objects: list[Vector[T]]=None):
        super().__init__(_objects)

    @staticmethod
    def debug():
        vectors = GenericVectorList[int]([
            Vector[int](1, 2), Vector[int](3, 4),
            Vector[int](5, 6), Vector[int](7, 8),
        ])                
        print(f"{vectors.to_dict()=}")
        print(f"{GenericVectorList[int].from_dict(vectors.to_dict()).to_dict()=}")
        assert vectors.to_dict() == GenericVectorList[int].from_dict(vectors.to_dict()).to_dict()
        assert vectors == GenericVectorList[int].from_dict(vectors.to_dict())
        print('Test passed')
