from __future__ import annotations
from pyevu import BBox2D, Vector2
from .._format import CocoBase, CocoBaseHandler

class Annotation(CocoBase):
    def __init__(
        self, id: int, image_id: int, category_id: int, segmentation: list[list[int]],
        area: float, bbox: tuple[int, int, int, int], iscrowd: int
    ):
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.area = area
        self.bbox = bbox
        self.iscrowd = iscrowd

    @property
    def bbox2d(self) -> BBox2D:
        return BBox2D(
            Vector2(*self.bbox[:2]),
            Vector2(*self.bbox[:2]) + Vector2(*self.bbox[2:])
        )

class Annotations(CocoBaseHandler[Annotation]):
    def __init__(self, _objects: list[Annotation] = None):
        super().__init__(_objects)

    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]

    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Annotations:
        return Annotations([Annotation.from_dict(val) for val in item_dict])


class Category(CocoBase):
    def __init__(self, id: int, name: str, supercategory: str):
        self.id = id
        self.name = name
        self.supercategory = supercategory


class Categories(CocoBaseHandler[Category]):
    def __init__(self, _objects: list[Category] = None):
        super().__init__(_objects)

    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]

    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Categories:
        return Categories([Category.from_dict(val) for val in item_dict])
