from __future__ import annotations
from typing import Callable
from datetime import datetime
from functools import partial
import os
import time
import copy
from tqdm import tqdm
import glob
from pyevu import BBox2D, Vector2
from ..vector import Vector
from ..polygon import Segmentation
from ..base import BaseHandler
from ._format import CocoBase, CocoBaseHandler, Info, Image, Images, License, Licenses
from ..labelme import Annotation as LabelmeAnn, \
    Shape as LabelmeShape, \
    Shapes as LabelmeShapes, \
    ShapeType as LabelmeShapeType

__all__ = [
    'Info', 'Image', 'Images', 'License', 'Licenses',
    'Annotation', 'Annotations',
    'Category', 'Categories',
    'Dataset',
    'BBoxResult', 'SegmentationResult', 'Result', 'Results'
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

    def to_labelme(
        self, annDir: str, overwrite: bool=False,
        showPbar: bool=False, leavePbar: bool=False,
        allowNoAnn: bool=False
    ):
        os.makedirs(annDir, exist_ok=True)
        if showPbar:
            pbar = tqdm(total=len(self.images), leave=leavePbar)
            pbar.set_description("Converting COCO to Labelme")
        for image in self.images:
            imgPath = image.file_name
            assert os.path.isfile(imgPath)
            if not os.path.isabs(imgPath):
                imgPath = os.path.abspath(imgPath)
            imgPath = os.path.relpath(imgPath, annDir)
            assert os.path.isfile(os.path.abspath(f"{annDir}/{imgPath}"))
            
            anns = self.annotations.search(lambda ann: ann.image_id == image.id)
            if len(anns) == 0 and not allowNoAnn:
                if showPbar:
                    pbar.update()
                continue
            labelmeAnn = LabelmeAnn(
                imagePath=imgPath,
                imageHeight=image.height,
                imageWidth=image.width
            )

            for ann in anns:
                category = self.categories.get(lambda cat: cat.id == ann.category_id)
                assert category is not None

                assert ann.bbox is not None
                if category.supercategory is not None:
                    label = f"{category.supercategory}/{category.name}"
                else:
                    label = category.name
                shape = LabelmeShape(
                    label=label,
                    points=[
                        Vector(*ann.bbox[:2]),
                        Vector(*ann.bbox[:2]) + Vector(*ann.bbox[2:])
                    ],
                    shape_type=LabelmeShapeType.rectangle
                )
                labelmeAnn.shapes.append(shape)
                if ann.segmentation is not None and len(ann.segmentation) > 0:
                    for poly in Segmentation.from_coco(ann.segmentation):
                        shape = LabelmeShape(
                            label=label,
                            points=list(poly),
                            shape_type=LabelmeShapeType.polygon
                        )
                        labelmeAnn.shapes.append(shape)

            annPath = annDir + '/' + os.path.basename(os.path.splitext(imgPath)[0]) + '.json'
            if os.path.isfile(annPath) and not overwrite:
                raise FileExistsError(annPath)
            labelmeAnn.save(annPath)
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()

    @classmethod
    def from_labelme(
        cls, annDir: str,
        showPbar: bool=False, leavePbar: bool=False,
        allowNoAnn: bool=False
    ) -> Dataset:
        assert os.path.isdir(annDir)
        paths = sorted(glob.glob(f"{annDir}/*.json"))
        dataset = Dataset()
        dataset.info.date_created = datetime.now()
        dataset.info.year = datetime.now().year
        dataset.info.description = "Converted from labelme."
        dataset.licenses.append(License(id=0, name='No License', url='N/A'))
        if showPbar:
            pbar = tqdm(total=len(paths), leave=leavePbar)
            pbar.set_description('Converting Labelme to COCO')
        for path in paths:
            labelmeAnn = LabelmeAnn.load(path)
            if len(labelmeAnn.shapes) == 0 and not allowNoAnn:
                if showPbar:
                    pbar.update()
                continue
            imgPath = labelmeAnn.imagePath
            if not os.path.isabs(imgPath):
                imgPath = os.path.abspath(f"{annDir}/{imgPath}")
            assert os.path.isfile(imgPath)
            image = Image(
                id=len(dataset.images),
                width=int(labelmeAnn.imageWidth),
                height=int(labelmeAnn.imageHeight),
                file_name=imgPath,
                license=0,
                date_captured=datetime.fromtimestamp(
                    time.mktime(time.gmtime(os.path.getctime(imgPath)))
                )
            )
            dataset.images.append(image)
            for shape in labelmeAnn.shapes:
                name = shape.label
                if '/' in name:
                    nameParts = name.split('/')
                    assert len(nameParts) == 2
                    supercategory = nameParts[0]
                    name = nameParts[1]
                    category = dataset.categories.get(
                        lambda cat: cat.supercategory == supercategory and cat.name == name
                    )
                else:
                    supercategory = 'labelme'
                    category = dataset.categories.get(lambda cat: cat.name == name)
                if category is None:
                    category = Category(
                        id=len(dataset.categories),
                        name=name,
                        supercategory=supercategory
                    )
                    dataset.categories.append(category)

                if shape.shape_type == LabelmeShapeType.rectangle:
                    assert len(shape.points) == 2
                    xmin = int(min([p.x for p in shape.points]))
                    ymin = int(min([p.y for p in shape.points]))
                    xmax = int(max([p.x for p in shape.points]))
                    ymax = int(max([p.y for p in shape.points]))
                    bbox = BBox2D(Vector2(xmin, ymin), Vector2(xmax, ymax))
                    ann = Annotation(
                        id=len(dataset.annotations),
                        image_id=image.id,
                        category_id=category.id,
                        segmentation=[],
                        area=bbox.area,
                        bbox=[bbox.v0.x, bbox.v0.y, bbox.xInterval.length, bbox.yInterval.length],
                        iscrowd=0
                    )
                    dataset.annotations.append(ann)
                else:
                    raise ValueError
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
        
        return dataset

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
        cleanupCat: bool=True, cleanupAnn: bool=True, cleanupImg: bool=True, cleanupLic: bool=True,
        reindex: bool=False,
        showPbar: bool=False, leavePbar: bool=False,
        applyToSelf: bool=False
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
        if catFilter is not None and cleanupAnn:
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
        if licFilter is not None and cleanupImg:
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
        if annFilter is not None or catFilter is not None and cleanupImg:
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
        if imgFilter is not None or annFilter is not None or catFilter is not None and cleanupLic:
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
        if imgFilter is not None or licFilter is not None and cleanupAnn:
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
        if annFilter is not None or imgFilter is not None or licFilter is not None and cleanupCat:
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

        showResult: bool = True
        resultColor: bool = (255, 0, 0)
        showResultLabel: bool = False
        showResultScore: bool = True

    def show_preview(self, results: Results=None):
        from ..util import CvUtil, PilUtil, MaskUtil, \
            VectorVar, ColorVar
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
                    
                if results is not None and s.showResult:
                    def draw_result_text(
                        img: np.ndarray, r: Result,
                        showLabel: bool=False, showScore: bool=True
                    ) -> np.ndarray:
                        if not showLabel and not showScore:
                            return img
                        if type(r) is BBoxResult:
                            bbox = BBox2D(
                                Vector2(*r.bbox[:2]), Vector2(*r.bbox[:2]) + Vector2(*r.bbox[2:]))
                        elif type(r) is SegmentationResult:
                            seg = Segmentation.from_coco(r.segmentation)
                            bbox = seg.bbox2d
                        else:
                            raise TypeError
                        
                        text = ''
                        if showLabel:
                            cat = self.categories.get(lambda cat: cat.id == r.category_id)
                            assert cat is not None
                            text += cat.name
                        if showScore:
                            if showLabel:
                                text += ' '
                            text += str(round(r.score, 2))
                        img = CvUtil.bbox_text(
                            img=img, text=text, bbox=bbox,
                            color=s.resultColor
                        )
                        return img

                    for r in results.search(lambda r: r.image_id == image.id):
                        cat = self.categories.get(lambda cat: cat.id == r.category_id)
                        assert cat is not None
                        if type(r) is BBoxResult:
                            if not s.showBBox or r.bbox is None:
                                continue
                            bbox = BBox2D(
                                Vector2(*r.bbox[:2]), Vector2(*r.bbox[:2]) + Vector2(*r.bbox[2:]))
                            img = CvUtil.rectangle(
                                img=img,
                                pt1=tuple(bbox.v0), pt2=tuple(bbox.v1),
                                color=s.resultColor,
                                thickness=2, lineType=cv2.LINE_AA
                            )
                        elif type(r) is SegmentationResult:
                            if not s.showSeg or r.segmentation is None:
                                continue
                            seg = Segmentation.from_coco(r.segmentation)
                            if not s.segIsTransparent:
                                img = cv2.drawContours(img, contours=seg.to_contours(
                                ), contourIdx=-1, color=s.segColor, thickness=-1)
                            else:
                                mask = np.zeros_like(img, dtype=np.uint8)
                                mask = cv2.drawContours(mask, contours=seg.to_contours(
                                ), contourIdx=-1, color=s.segColor, thickness=-1)
                                img = cv2.addWeighted(
                                    src1=img, src2=mask, alpha=1, beta=1, gamma=0)
                        else:
                            raise TypeError
                        if s.showResultLabel or s.showResultScore:
                            img = draw_result_text(
                                img=img, r=r,
                                showLabel=s.showResultLabel, showScore=s.showResultScore
                            )

                vis.show(img, title=f'image.id={image.id}, filename={os.path.basename(image.file_name)}')

class BBoxResult(CocoBase):
    def __init__(
        self, image_id: int, category_id: int,
        bbox: list[int, int, int, int],
        score: float
    ):
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = bbox
        self.score = score

    @property
    def bbox2d(self) -> BBox2D:
        return BBox2D(
            Vector2(*self.bbox[:2]),
            Vector2(*self.bbox[:2]) + Vector2(*self.bbox[2:])
        )

class SegmentationResult(CocoBase):
    def __init__(
        self, image_id: int, category_id: int,
        segmentation: list[list[int]],
        score: float
    ):
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.score = score
    
    @property
    def seg(self) -> Segmentation:
        return Segmentation.from_coco(self.segmentation)

Result = BBoxResult | SegmentationResult

class Results(BaseHandler[Result]):
    def __init__(self, _objects: list[Result]=None):
        super().__init__(_objects)

    def to_dict(self, compressed: bool = True, **kwargs) -> dict:
        return [obj.to_dict() for obj in self]

    @classmethod
    def from_dict(cls, item_dict: dict) -> Results:
        def from_result(result_dict: dict) -> Result:
            if 'bbox' in result_dict:
                return BBoxResult.from_dict(result_dict)
            elif 'segmentation' in result_dict:
                return SegmentationResult.from_dict(result_dict)
            else:
                raise KeyError
        
        assert type(item_dict) is list
        return Results([from_result(val) for val in item_dict])

    def to_annotations(self, minPairingIoU: float=0.5) -> Annotations:
        annotations = Annotations()
        image_ids = sorted(list(set([r.image_id for r in self])))
        category_ids = sorted(list(set([r.category_id for r in self])))
        for image_id in image_ids:
            for category_id in category_ids:
                results = self.search(lambda r: r.image_id == image_id and r.category_id == category_id)
                if len(results) == 0:
                    continue
                
                bboxResults = results.search(lambda r: type(r) is BBoxResult)
                segResults = results.search(lambda r: type(r) is SegmentationResult)
                pairs: list[tuple[BBoxResult | None, SegmentationResult | None]] = []
                for segResult in segResults:
                    segBbox = segResult.seg.bbox2d
                    bestIoU = None
                    bestIdx = None
                    for idx in range(len(bboxResults)):
                        iou = BBox2D.IoU(segBbox, bboxResults[idx].bbox2d)
                        if iou >= minPairingIoU and (bestIoU is None or iou > bestIoU):
                            bestIoU = iou
                            bestIdx = idx
                    if bestIdx is not None:
                        pairs.append((bboxResults[bestIdx], segResult))
                        del bboxResults[bestIdx]
                    else:
                        pairs.append((None, segResult))
                for bboxResult in bboxResults:
                    pairs.append((bboxResult, None))

                for bboxResult, segResult in pairs:
                    assert bboxResult is not None or segResult is not None
                    segmentation = segResult.segmentation if segResult is not None else None
                    bbox = bboxResult.bbox if bboxResult is not None else segResult.seg.bbox
                    bbox2d = BBox2D(Vector2(*bbox[:2]), Vector2(*bbox[:2]) + Vector2(*bbox[2:]))
                    ann = Annotation(
                        id=len(annotations),
                        image_id=image_id, category_id=category_id,
                        segmentation=segmentation,
                        area=bbox2d.area,
                        bbox=bbox, iscrowd=0
                    )
                    annotations.append(ann)
        return annotations
