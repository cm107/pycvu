from __future__ import annotations
from pyevu import BBox2D, Vector2, Quad2
from .._format import CocoBase, CocoBaseHandler, \
    AnnotationBase, AnnotationsBase, \
    CategoryBase, CategoriesBase
import numpy as np

class Annotation(AnnotationBase):
    def __init__(
        self, id: int, image_id: int, category_id: int,
        text: str, quad: list[float],
        score: float=None
    ):
        super().__init__(
            id=id, image_id=image_id, category_id=category_id,
            score=score
        )
        self.text = text
        self.quad = quad

    @property
    def quad2d(self) -> Quad2:
        return Quad2.from_numpy(np.array(self.quad).reshape(-1, 2))

    @property
    def bbox2d(self) -> BBox2D:
        return self.quad2d.bbox2d

class Annotations(AnnotationsBase[Annotation]):
    def __init__(self, _objects: list[Annotation] = None):
        super().__init__(_objects)

    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Annotations:
        return Annotations([Annotation.from_dict(val) for val in item_dict])


class Category(CategoryBase):
    def __init__(self, id: int, name: str, supercategory: str):
        super().__init__(
            id=id, name=name, supercategory=supercategory
        )

class Categories(CategoriesBase[Category]):
    def __init__(self, _objects: list[Category] = None):
        super().__init__(_objects)

    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Categories:
        return Categories([Category.from_dict(val) for val in item_dict])
