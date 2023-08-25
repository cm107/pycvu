from __future__ import annotations
import os
from typing import TYPE_CHECKING

import cv2
from tqdm import tqdm
from ....create_ml import CreateML
if TYPE_CHECKING:
    from . import Dataset
from .._structs import Annotation, Category
from ..._format import Image, License

def to_createml(self: Dataset, showPbar: bool=False) -> CreateML.ObjectDetection.Dataset:
    imgIdToImg: dict[int, Image] = {}
    for image in self.images:
        imgIdToImg[image.id] = image
    imgIdToAnns: dict[int, list[Annotation]] = {}
    for ann in self.annotations:
        if ann.image_id not in imgIdToAnns:
            imgIdToAnns[ann.image_id] = [ann]
        else:
            imgIdToAnns[ann.image_id].append(ann)
    
    _frames: list[CreateML.ObjectDetection.Frame] = []

    imgDir: str = None
    vals = imgIdToAnns.items()
    if showPbar:
        vals = tqdm(vals, desc="Converting to CreateML", unit='frame(s)')
    for imgId, anns in vals:
        image = imgIdToImg[imgId]
        
        _imgDir = os.path.dirname(image.file_name)
        if imgDir is None:
            imgDir = _imgDir
        elif imgDir != _imgDir:
            raise Exception(f"Inconsistent image directory: '{imgDir}' and '{_imgDir}'")

        _imagefilename = os.path.basename(image.file_name)
        _anns: list[CreateML.ObjectDetection.Annotation] = []
        for ann in anns:
            bbox = ann.bbox2d
            _x, _y = tuple(bbox.center)
            _width = bbox.xInterval.length
            _height = bbox.yInterval.length

            cat = self.categories.get(lambda cat: cat.id == ann.category_id)
            _label = cat.name

            _ann = CreateML.ObjectDetection.Annotation(
                coordinates=CreateML.ObjectDetection.Coordinates(
                    x=_x, y=_y, width=_width, height=_height
                ),
                label=_label
            )
            _anns.append(_ann)
        
        _frame = CreateML.ObjectDetection.Frame(
            imagefilename=_imagefilename,
            annotation=_anns
        )
        _frames.append(_frame)

    return CreateML.ObjectDetection.Dataset(_frames)

@classmethod
def from_createml(
    cls: type[Dataset], ds: CreateML.ObjectDetection.Dataset, imgDir: str,
    showPbar: bool=False
) -> Dataset:
    dataset = cls()
    dataset.licenses.append(License(id=0, name='MIT', url='N/A'))
    if not os.path.isdir(imgDir):
        raise FileNotFoundError
    
    vals = ds.frames
    if showPbar:
        vals = tqdm(vals, desc="Converting to COCO", unit='frame(s)')
    for frame in vals:
        imgPath = f"{imgDir}/{frame.imagefilename}"
        if not os.path.isfile(imgPath):
            raise FileNotFoundError
        img = cv2.imread(imgPath)
        height, width = img.shape[:2]
        image = Image(
            id=len(dataset.images),
            width=width, height=height,
            file_name=frame.imagefilename,
            license=0
        )
        dataset.images.append(image)
        for _ann in frame.annotation:
            cat = dataset.categories.get(lambda _cat: _cat.name == _ann.label)
            if cat is None:
                cat = Category(
                    id=len(dataset.categories),
                    name=_ann.label, supercategory='coreml'
                )
                dataset.categories.append(cat)
            
            w = _ann.coordinates.width
            h = _ann.coordinates.height
            xmin = _ann.coordinates.x - 0.5 * w
            ymin = _ann.coordinates.y - 0.5 * h
            ann = Annotation(
                id=len(dataset.annotations),
                image_id=image.id,
                category_id=cat.id,
                segmentation=[],
                area=w * h,
                bbox=[xmin, ymin, w, h],
                iscrowd=0
            )
            dataset.annotations.append(ann)
    return dataset
