from __future__ import annotations
from typing import TYPE_CHECKING, Type, Callable
if TYPE_CHECKING:
    from .._artist import Artist

import os
from tqdm import tqdm
from datetime import datetime
from shutil import rmtree
import cv2
import numpy as np
import multiprocessing as mp
from itertools import repeat
from ...coco.object_detection import *
from ...mask import MaskHandler

def update_dataset(i: int, dumpDir: str, dataset: Dataset, result: np.ndarray, maskHandler: MaskHandler):
    imgPath = f"{dumpDir}/frame{i}.png"
    cv2.imwrite(imgPath, result)
    image = Image(
        id=len(dataset.images),
        license=dataset.licenses[0].id,
        width=result.shape[1], height=result.shape[0],
        file_name=imgPath, date_captured=datetime.now()
    )
    dataset.images.append(image)

    for j, mask in enumerate(maskHandler):
        if mask._mask.sum() == 0:
            continue

        category = dataset.categories.get(
            lambda c: c.name == mask.setting.category
            and c.supercategory == mask.setting.supercategory
        )
        if category is None:
            category = Category(
                id=len(dataset.categories),
                name=mask.setting.category,
                supercategory=mask.setting.supercategory
            )
            dataset.categories.append(category)
        
        bbox = mask.bbox
        seg = mask.segmentation
        seg = seg.prune(lambda poly: len(poly) < 3)
        ann = Annotation(
            id=len(dataset.annotations),
            image_id=image.id, category_id=category.id,
            segmentation=seg.coco,
            area=bbox.area,
            bbox=[
                bbox.v0.x, bbox.v0.y,
                bbox.xInterval.length, bbox.yInterval.length
            ],
            iscrowd=0
        )
        dataset.annotations.append(ann)

def draw_and_get_masks(
    self: Artist,
    resultCallback: Callable[[np.ndarray, MaskHandler], None]=None
) -> tuple[np.ndarray, MaskHandler]:
    result, maskHandler = self.draw_and_get_masks()
    if resultCallback is not None:
        resultCallback(result, maskHandler)
    return result, maskHandler

def _draw_and_get_masks(
    frame: int,
    self: Artist,
    resultCallback: Callable[[np.ndarray, MaskHandler], None]=None
) -> tuple[int, np.ndarray, MaskHandler]:
    result, maskHandler = draw_and_get_masks(self, resultCallback)
    return frame, result, maskHandler

def _generate_dataset_object_detection(
    self: Artist, frames: int, dumpDir: str="artist_dataset_dump",
    showPbar: bool=True,
    resultCallback: Callable[[np.ndarray, MaskHandler], None]=None,
    batchSize: int=1, saveCpus: int=2
):
    if os.path.isdir(dumpDir):
        rmtree(dumpDir)
    os.makedirs(dumpDir, exist_ok=True)

    dataset = Dataset()
    dataset.info.description = 'Generated with pycvu Artist.'
    dataset.info.date_created = datetime.now()
    dataset.info.year = datetime.now().year
    dataset.licenses.append(
        License(id=0, name='No License', url='N/A')
    )
    if batchSize == 1:
        frameList = list(range(frames))
        if showPbar:
            frameList = tqdm(frameList, leave=True, desc="Generating Dataset")
        for i in frameList:
            result, maskHandler = draw_and_get_masks(self, resultCallback)
            update_dataset(
                i=i, dumpDir=dumpDir, dataset=dataset,
                result=result, maskHandler=maskHandler
            )
    else:
        _frameList = list(range(frames))
        if showPbar:
            pbar = tqdm(total=frames, leave=True, desc="Generating Dataset")
        frameListBatches: list[list[int]] = [_frameList[i*batchSize:(i+1)*batchSize] for i in range((len(_frameList)+batchSize-1)//batchSize)]
        
        for frameList in frameListBatches:
            allocatedCpus = mp.cpu_count() - saveCpus
            pool = mp.Pool(allocatedCpus)
            data = pool.starmap(_draw_and_get_masks, zip(frameList, repeat(self), repeat(resultCallback)))
            pool.close()
            for i, result, maskHandler in data:
                update_dataset(
                    i=i, dumpDir=dumpDir, dataset=dataset,
                    result=result, maskHandler=maskHandler
                )
                pbar.update()
    dataset.save(f"{dumpDir}/dataset.json")