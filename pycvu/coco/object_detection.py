from __future__ import annotations
from typing import Callable
from datetime import datetime
import copy
from tqdm import tqdm
from ._format import CocoBase, CocoBaseHandler, Info, Image, Images, License, Licenses

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
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.area = area
        self.bbox = bbox
        self.iscrowd = iscrowd


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


class Dataset(CocoBase):
    def __init__(
        self, info: Info = None, images: Images = None, licenses: Licenses = None,
        annotations: Annotations = None,
        categories: Categories = None
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

    @classmethod
    def combine(cls, sources: list[Dataset | str], showPbar: bool=False) -> Dataset:
        combined = Dataset()
        
        # Process Info
        combined.info.description = "Combined dataset using pycvu."
        combined.info.date_created = datetime.now()
        combined.info.year = datetime.now().year

        outerPbar = tqdm(total=len(sources), leave=True) if showPbar else None
        if showPbar:
            outerPbar.set_description("Combining Datasets")
        for src in sources:
            if type(src) is str:
                dataset = Dataset.load(src)
            elif type(src) is Dataset:
                dataset = src
            else:
                raise TypeError
            
            innerPbar = tqdm(total=len(dataset.images), leave=False) if showPbar else None
            dataset.images.sort(lambda image: image.id)
            for image in dataset.images:
                # Process License
                license = dataset.licenses.get(lambda lic: lic.id == image.license)
                assert license is not None
                lic = combined.licenses.get(
                    lambda lic: all([
                        getattr(lic, key) == getattr(license, key)
                        for key in license.__dict__
                        if key != 'id'
                    ])
                )
                if lic is None:
                    # New license.
                    lic = license.copy()
                    lic.id = len(combined.licenses)
                    combined.licenses.append(lic)

                # Process Image
                img = image.copy()
                img.id = len(combined.images)
                img.license = lic.id
                combined.images.append(img)

                for annotation in dataset.annotations.search(lambda ann: ann.image_id == image.id):
                    # Process Category
                    category = dataset.categories.get(lambda cat: cat.id == annotation.category_id)
                    assert category is not None
                    cat = combined.categories.get(
                        lambda cat: all([
                            getattr(cat, key) == getattr(category, key)
                            for key in category.__dict__
                            if key != 'id'
                        ])
                    )
                    if cat is None:
                        # New category.
                        cat = category.copy()
                        cat.id = len(combined.categories)
                        combined.categories.append(cat)

                    # Process Annotation
                    ann = annotation.copy()
                    ann.id = len(combined.annotations)
                    ann.image_id = img.id
                    ann.category_id = cat.id
                    combined.annotations.append(ann)
                if showPbar:
                    innerPbar.update()
            if showPbar:
                innerPbar.close()
                outerPbar.update()
        if showPbar:
            outerPbar.close()
        return combined

    def filter(
        self,
        catFilter: Callable[[Category], bool]=None,
        annFilter: Callable[[Annotation], bool]=None,
        imgFilter: Callable[[Image], bool]=None,
        licFilter: Callable[[License], bool]=None,
        reindex: bool=False, # TODO
        showPbar: bool=False, leavePbar: bool=False,
        applyToSelf: bool=False # TODO
    ) -> Dataset:
        # TODO: docstring
        # Note: Each filter affects all handlers either directly or through broken references.
        # Example: catFilter affects categories directly, and affects annotations, images, and
        #          licenses through broken references.
        if catFilter is None and annFilter is None and imgFilter is None and licFilter is None:
            raise ValueError("Must specify at least one filter callback.")
        
        # Apply Filters (Easy Part)
        if not applyToSelf:
            if showPbar:
                pbar = tqdm(total=1, leave=leavePbar)
                pbar.set_description("Copying Data For Filtering")
            result = copy.deepcopy(self)
            if showPbar:
                pbar.update()
                pbar.close()
        else:
            result = self
        for handler, callback in [
            (result.categories, catFilter),
            (result.annotations, annFilter),
            (result.images, imgFilter),
            (result.licenses, licFilter)
        ]:
            if callback is None:
                continue
            handler: CocoBaseHandler[CocoBase] = handler
            callback: Callable[[CocoBase], bool] = callback
            if callback is not None:
                if showPbar:
                    pbar = tqdm(total=len(handler), leave=leavePbar)
                    pbar.set_description(f"Filtering {type(handler).__name__}")
                for idx in list(range(len(handler)))[::-1]:
                    obj = handler[idx]
                    if not callback(obj):
                        del handler[idx]
                    if showPbar:
                        pbar.update()
                if showPbar:
                    pbar.close()

        # Cleanup Broken References (Hard Part)
        if catFilter is not None:
            # Categories affect Annotations
            if showPbar:
                pbar = tqdm(total=len(result.annotations), leave=leavePbar)
                pbar.set_description("Cleanup Annotations w/o Categories")
            for idx in list(range(len(result.annotations)))[::-1]:
                ann = result.annotations[idx]
                if result.categories.get(id=ann.category_id) is None:
                    del result.annotations[idx]
                if showPbar:
                    pbar.update()
            if showPbar:
                pbar.close()
        if licFilter is not None:
            # Licenses affect Images
            if showPbar:
                pbar = tqdm(total=len(result.images), leave=leavePbar)
                pbar.set_description("Cleanup Images w/o Licenses")
            for idx in list(range(len(result.images)))[::-1]:
                img = result.images[idx]
                if result.licenses.get(id=img.license) is None:
                    del result.images[idx]
                if showPbar:
                    pbar.update()
            if showPbar:
                pbar.close()
        if annFilter is not None or catFilter is not None:
            # Annotations affect Images -> Images Done
            if showPbar:
                pbar = tqdm(total=len(result.images), leave=leavePbar)
                pbar.set_description("Cleanup Images w/o Annotations")
            for idx in list(range(len(result.images)))[::-1]:
                img = result.images[idx]
                if result.annotations.get(image_id=img.id) is None:
                    del result.images[idx]
                if showPbar:
                    pbar.update()
            if showPbar:
                pbar.close()
        if imgFilter is not None or annFilter is not None or catFilter is not None:
            # Images affect Licenses -> Licenses Done
            if showPbar:
                pbar = tqdm(total=len(result.licenses), leave=leavePbar)
                pbar.set_description("Cleanup Licenses w/o Images")
            for idx in list(range(len(result.licenses)))[::-1]:
                lic = result.licenses[idx]
                if result.images.get(license=lic.id) is None:
                    del result.licenses[idx]
                if showPbar:
                    pbar.update()
            if showPbar:
                pbar.close()
        if imgFilter is not None or licFilter is not None:
            # Images affect Annotations -> Annotations Done
            if showPbar:
                pbar = tqdm(total=len(result.annotations), leave=leavePbar)
                pbar.set_description("Cleanup Annotations w/o Images")
            for idx in list(range(len(result.annotations)))[::-1]:
                ann = result.annotations[idx]
                if result.images.get(id=ann.image_id) is None:
                    del result.annotations[idx]
                if showPbar:
                    pbar.update()
            if showPbar:
                pbar.close()
        if annFilter is not None or imgFilter is not None or licFilter is not None:
            # Annotations affect Categories -> Categories Done
            if showPbar:
                pbar = tqdm(total=len(result.categories), leave=leavePbar)
                pbar.set_description("Cleanup Categories w/o Annotations")
            for idx in list(range(len(result.categories)))[::-1]:
                cat = result.categories[idx]
                if result.annotations.get(category_id=cat.id) is None:
                    del result.categories[idx]
                if showPbar:
                    pbar.update()
            if showPbar:
                pbar.close()

        if reindex:
            result.reindex(applyToSelf=True, showPbar=showPbar, leavePbar=leavePbar)

        return result
    
    def reindex(
        self, applyToSelf: bool=True,
        showPbar: bool=False, leavePbar: bool=False
    ) -> Dataset:
        if not applyToSelf:
            result = copy.deepcopy(self)
        else:
            result = self
        
        def updateLicenseIdInImages(oldId: int, newId: int):
            for img in result.images.search(lambda img: img.license == oldId):
                img.license = newId

        def updateImgIdInAnnotations(oldId: int, newId: int):
            for ann in result.annotations.search(lambda ann: ann.image_id == oldId):
                ann.image_id = newId
        
        def updateCatIdInAnnotations(oldId: int, newId: int):
            for ann in result.annotations.search(lambda ann: ann.category_id == oldId):
                ann.category_id = newId

        for handler, idUpdateCallback in [
            (result.licenses, updateLicenseIdInImages),
            (result.images, updateImgIdInAnnotations),
            (result.categories, updateCatIdInAnnotations),
            (result.annotations, None)
        ]:
            handler: CocoBaseHandler = handler
            handler.reindex(
                showPbar=showPbar, leavePbar=leavePbar,
                applyToSelf=True,
                idUpdateCallback=idUpdateCallback
            )
        
        return result

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
        from ..vis.cv import SimpleVisualizer

        s = Dataset.PreviewSettings

        vis = SimpleVisualizer()
        with vis.loop(self.images) as loop:
            while not loop.done:
                image = self.images[loop.index]
                img = cv2.imread(image.file_name)
                assert img is not None

                anns = self.annotations.search(
                    lambda ann: ann.image_id == image.id)
                for ann in anns:
                    bbox = BBox2D(
                        Vector2(*ann.bbox[:2]), Vector2(*ann.bbox[:2]) + Vector2(*ann.bbox[2:]))
                    seg = Segmentation.from_coco(ann.segmentation)

                    if seg is not None and s.showSeg:
                        if not s.segIsTransparent:
                            img = cv2.drawContours(img, contours=seg.to_contours(
                            ), contourIdx=-1, color=s.segColor, thickness=-1)
                        else:
                            mask = np.zeros_like(img, dtype=np.uint8)
                            mask = cv2.drawContours(mask, contours=seg.to_contours(
                            ), contourIdx=-1, color=s.segColor, thickness=-1)
                            img = cv2.addWeighted(
                                src1=img, src2=mask, alpha=1, beta=1, gamma=0)

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
                        img = CvUtil.text(img, text=category.name, org=tuple(
                            bbox.center), color=s.labelColor)

                vis.show(img, title=f'image.id={image.id}')
