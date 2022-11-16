from __future__ import annotations
from ..base import BaseHandler
from ._format import CocoBase, Info, Image, Images, License, Licenses

__all__ = [
    'Info', 'Image', 'Images', 'License', 'Licenses',
    'Annotation', 'Annotations',
    'Category', 'Categories',
    'Dataset'
]

class Annotation(CocoBase):
    def __init__(
        self, id: int, image_id: int, category_id: int, segmentation: list[list[int]],
        area: float, bbox: tuple[int, int, int, int], iscrowd: int
    ):
        self.id = id; self.image_id = image_id; self.category_id = category_id
        self.segmentation = segmentation; self.area = area; self.bbox = bbox
        self.iscrowd = iscrowd

class Annotations(BaseHandler[Annotation]):
    def __init__(self, _objects: list[Annotation]=None):
        super().__init__(_objects)

    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Annotations:
        return Annotations([Annotation.from_dict(val) for val in item_dict])

class Category(CocoBase):
    def __init__(self, id: int, name: str, supercategory: str):
        self.id = id; self.name = name; self.supercategory = supercategory

class Categories(BaseHandler[Category]):
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

    class PreviewSettings:
        showBBox: bool = True
        bboxColor: tuple = (0, 0, 255)
        
        showSeg: bool = True
        segIsTransparent: bool = True
        segColor: tuple = (255, 255, 0)
        
        showLabel: bool = True
        labelColor: tuple = (0, 255, 0)

    def show_preview(self):
        from ..util import CvUtil, PilUtil
        from ..polygon import Segmentation
        from pyevu import BBox2D, Vector2
        import cv2
        import numpy as np

        s = Dataset.PreviewSettings

        for image in self.images:
            img = cv2.imread(image.file_name)
            assert img is not None

            anns = self.annotations.search(lambda ann: ann.image_id == image.id)
            for ann in anns:
                bbox = BBox2D(Vector2(*ann.bbox[:2]), Vector2(*ann.bbox[:2]) + Vector2(*ann.bbox[2:]))
                seg = Segmentation.from_coco(ann.segmentation)
                
                if seg is not None and s.showSeg:
                    if not s.segIsTransparent:
                        img = cv2.drawContours(img, contours=seg.to_contours(), contourIdx=-1, color=s.segColor, thickness=-1)
                    else:
                        mask = np.zeros_like(img, dtype=np.uint8)
                        mask = cv2.drawContours(mask, contours=seg.to_contours(), contourIdx=-1, color=s.segColor, thickness=-1)
                        img = cv2.addWeighted(src1=img, src2=mask, alpha=1, beta=1, gamma=0)

                if bbox is not None and s.showBBox:
                    img = CvUtil.rectangle(
                        img=img,
                        pt1=tuple(bbox.v0), pt2=tuple(bbox.v1),
                        color=s.bboxColor,
                        thickness=2, lineType=cv2.LINE_AA
                    )
                
                if s.showLabel:
                    category = self.categories.get(id=ann.category_id)
                    assert category is not None
                    img = CvUtil.text(img, text=category.name, org=tuple(bbox.center), color=s.labelColor)
            
            cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('preview', 500, 500)
            cv2.imshow('preview', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
