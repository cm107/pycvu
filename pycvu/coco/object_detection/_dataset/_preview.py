from __future__ import annotations
import cv2
import numpy as np
from pyevu import BBox2D, Vector2
from ....polygon import Segmentation
from ....util import CvUtil
from .._result import BBoxResult, SegmentationResult, Result, Results

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Dataset

from ...dataset_base._preview import *

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
