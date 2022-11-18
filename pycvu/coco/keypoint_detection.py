from __future__ import annotations
from . import object_detection as parent
from ._format import CocoBase, CocoBaseHandler, Info, Image, Images, License, Licenses

__all__ = [
    'Info', 'Image', 'License', 'Licenses',
    'Annotation', 'Annotations',
    'Category', 'Categories',
    'Dataset'
]

class Annotation(parent.Annotation):
    def __init__(
        self, id: int, image_id: int, category_id: int, segmentation: list[int],
        area: float, bbox: tuple[int, int, int, int], iscrowd: int,
        keypoints: list[int], num_keypoints: int
    ):
        super().__init__(
            id=id, image_id=image_id, category_id=category_id,
            segmentation=segmentation, area=area, bbox=bbox,
            iscrowd=iscrowd
        )
        self.keypoints = keypoints
        self.num_keypoints = num_keypoints

class Annotations(CocoBaseHandler[Annotation]):
    def __init__(self, _objects: list[Annotation]=None):
        super().__init__(_objects)

    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Annotations:
        return Annotations([Annotation.from_dict(val) for val in item_dict])

class Category(parent.Category):
    def __init__(
        self, id: int, name: str, supercategory: str,
        keypoints: list[str], skeleton: list[int]
    ):
        super().__init__(id=id, name=name, supercategory=supercategory)
        self.keypoints = keypoints
        self.skeleton = skeleton

class Categories(CocoBaseHandler[Category]):
    def __init__(self, _objects: list[Category]=None):
        super().__init__(_objects)

    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Categories:
        return Categories([Category.from_dict(val) for val in item_dict])

class Dataset(CocoBase):
    def __init__(
        self, info: Info=None, images: Images=None, licenses: Licenses=None,
        annotations: Annotations=None,
        categories: Categories=None
    ):
        self.info = info if info is not None else Info()
        self.images = images if images is not None else Images()
        self.licenses = licenses if licenses is not None else Licenses()
        self.annotations = annotations if annotations is not None else Annotations()
        self.categories = categories if categories is not None else Categories()

    def to_dict(self) -> dict:
        return {key: val.to_dict() for key, val in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Dataset:
        return Dataset(
            info=Info.from_dict(item_dict['info']),
            images=Images.from_dict(item_dict['images']),
            licenses=Licenses.from_dict(item_dict['licenses']),
            annotations=Annotations.from_dict(item_dict['annotations']),
            categories=Categories.from_dict(item_dict['categories'])
        )
