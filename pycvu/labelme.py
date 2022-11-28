from __future__ import annotations
from typing import Any
from enum import Enum
from .vector import Vector
from .base import Base, BaseHandler

class ShapeType(Enum):
    rectangle = "rectangle"
    polygon = "polygon"

class Shape(Base):
    def __init__(
        self, label: str, points: list[Vector],
        group_id: int | None=None,
        shape_type: ShapeType=ShapeType.rectangle,
        flags: dict={}
    ):
        self.label = label
        self.points = points
        self.group_id = group_id
        self.shape_type = shape_type
        self.flags = flags

    def to_dict(self) -> dict:
        return dict(
            label=self.label,
            points=[list(p) for p in self.points],
            group_id=self.group_id,
            shape_type=self.shape_type.name,
            flags=self.flags
        )
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Shape:
        return Shape(
            label=item_dict['label'],
            points=[Vector(*p) for p in item_dict['points']],
            group_id=item_dict['group_id'],
            shape_type=ShapeType[item_dict['shape_type']],
            flags=item_dict['flags']
        )

class Shapes(BaseHandler[Shape]):
    def __init__(self, _objects: list[Shape]=None):
        super().__init__(_objects)

class Annotation(Base):
    def __init__(
        self,
        version: str='5.1.1', flags: dict={},
        shapes: Shapes=None,
        imagePath: str="", imageData: Any | None=None,
        imageHeight: int=-1, imageWidth: int=-1
    ):
        self.version = version
        self.shapes = shapes if shapes is not None else Shapes()
        self.imagePath = imagePath
        self.imageData = imageData
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth

    def to_dict(self) -> dict:
        item_dict = self.__dict__
        item_dict['shapes'] = [shape.to_dict() for shape in item_dict['shapes']]
        return item_dict
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Annotation:
        params = item_dict.copy()
        params['shapes'] = [Shape.from_dict(shape) for shape in params['shapes']]
        return Annotation(**params)
