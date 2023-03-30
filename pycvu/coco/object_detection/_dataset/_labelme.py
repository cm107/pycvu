from __future__ import annotations
from datetime import datetime
import os
import time
from tqdm import tqdm
import glob
from pyevu import BBox2D, Vector2
from ....vector import Vector
from ....polygon import Segmentation
from ....labelme import Annotation as LabelmeAnn, \
    Shape as LabelmeShape, \
    ShapeType as LabelmeShapeType
from ..._format import Image, License
from .._structs import Annotation, Category

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Dataset

def to_labelme(
    self: Dataset, annDir: str, overwrite: bool=False,
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
    cls: type[Dataset], annDir: str,
    showPbar: bool=False, leavePbar: bool=False,
    allowNoAnn: bool=False,
    checkImageExists: bool=True
) -> Dataset:
    assert os.path.isdir(annDir)
    paths = sorted(glob.glob(f"{annDir}/*.json"))
    dataset = cls()
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
        if checkImageExists:
            assert os.path.isfile(imgPath)
        image = Image(
            id=len(dataset.images),
            width=int(labelmeAnn.imageWidth),
            height=int(labelmeAnn.imageHeight),
            file_name=imgPath,
            license=0,
            date_captured=datetime.fromtimestamp(
                time.mktime(time.gmtime(os.path.getctime(imgPath)))
            ) if os.path.isfile(imgPath) else None
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
