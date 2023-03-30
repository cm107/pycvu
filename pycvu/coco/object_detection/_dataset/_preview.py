from __future__ import annotations
import os
import cv2
import numpy as np
from tqdm import tqdm
from pyevu import BBox2D, Vector2
from ....polygon import Segmentation
from ....util import CvUtil
from ....vis.cv import SimpleVisualizer
from .._result import BBoxResult, SegmentationResult, Result, Results

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Dataset
    from .. import Image

class PreviewSettings:
    showGT: bool = True
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

def draw_preview(self: Dataset, img: np.ndarray, image_id: int, results: Results=None) -> np.ndarray:
    s = PreviewSettings
    if s.showGT:
        anns = self.annotations.search(
            lambda ann: ann.image_id == image_id)
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
        
    if s.showResult:
        if results is None:
            results = Results()
            for ann in self.annotations:
                if ann.score is None:
                    continue
                
                results.append(
                    BBoxResult(
                        image_id=ann.image_id,
                        category_id=ann.category_id,
                        bbox=ann.bbox,
                        score=ann.score
                    )
                )
                if ann.segmentation is not None and len(ann.segmentation) > 0:
                    results.append(
                        SegmentationResult(
                            image_id=ann.image_id,
                            category_id=ann.category_id,
                            segmentation=ann.segmentation,
                            score=ann.score
                        )
                    )

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

        for r in results.search(lambda r: r.image_id == image_id):
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
    return img

def get_preview_from_image(
    self: Dataset, image: Image, results: Results=None,
    imgDir: str=None
) -> np.ndarray:
    if imgDir is not None:
        if not os.path.isdir(imgDir):
            raise FileNotFoundError
        path = f"{imgDir}/{image.file_name}"
    else:
        path = image.file_name
    if not os.path.isfile(path):
        raise FileNotFoundError
    img = cv2.imread(path)
    assert img is not None
    img = self.draw_preview(
        img=img, image_id=image.id,
        results=results
    )
    return img

def get_preview_from_image_idx(
    self: Dataset, imageIdx: int, results: Results=None,
    imgDir: str=None
) -> np.ndarray:
    image = self.images[imageIdx]
    return self.get_preview_from_image(
        image=image, results=results, imgDir=imgDir
    )

def get_preview(
    self: Dataset, image: Image | int, results: Results=None,
    imgDir: str=None
) -> np.ndarray:
    if type(image) is int:
        image = self.images[imageIdx]
    return self.get_preview_from_image(
        image=image, results=results, imgDir=imgDir
    )

def show_preview(
    self: Dataset, results: Results=None,
    imgDir: str=None
):
    vis = SimpleVisualizer()
    with vis.loop(self.images) as loop:
        while not loop.done:
            image = self.images[loop.index]
            img = self.get_preview_from_image_idx(loop.index, results=results, imgDir=imgDir)
            vis.show(img, title=f'image.id={image.id}, filename={os.path.basename(image.file_name)}')

def show_filename(
    self: Dataset, filename: str,
    results: Results=None, imgDir: str=None
):
    image = self.images.get(lambda img: os.path.basename(img.file_name) == filename)
    if image is None:
        raise FileNotFoundError(f"Failed to find image in dataset with a filename of {filename}")
    img = self.get_preview_from_image(image, results=results, imgDir=imgDir)
    vis = SimpleVisualizer()
    vis.show(img, title=f"filename={os.path.basename(image.file_name)}")

def save_preview(
    self: Dataset, saveDir: str,
    results: Results=None, imgDir: str=None, showPbar: bool=False
):
    os.makedirs(saveDir, exist_ok=True)
    if showPbar:
        pbar = tqdm(total=len(self.images), leave=False)
    for i, image in enumerate(self.images):
        img = self.get_preview_from_image_idx(i, results=results, imgDir=imgDir)
        filename = os.path.basename(image.file_name)
        savePath = f"{saveDir}/{filename}"
        cv2.imwrite(savePath, img)
        if showPbar:
            pbar.update()
    if showPbar:
        pbar.close()

def save_filename(
    self: Dataset, filename: str, savePath: str,
    results: Results=None, imgDir: str=None
):
    image = self.images.get(lambda img: os.path.basename(img.file_name) == filename)
    if image is None:
        raise FileNotFoundError(f"Failed to find image in dataset with a filename of {filename}")
    img = self.get_preview_from_image(image, results=results, imgDir=imgDir)
    cv2.imwrite(savePath, img)
