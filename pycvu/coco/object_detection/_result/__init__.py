from __future__ import annotations
from pyevu import BBox2D, Vector2
from ....polygon import Segmentation
from ....base import BaseHandler
from ..._format import CocoBase
from .._structs import Annotation, Annotations

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
    
    @property
    def bbox2d(self) -> BBox2D:
        return self.seg.bbox2d

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

    @classmethod
    def from_annotations(cls, anns: Annotations) -> Results:
        results = Results()
        for ann in anns:
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
        return results

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

    # Note: These two methods are deprecated.
    # They will likely be deleted in the future.
    from ._old_eval import gtdt_match
    from ._old_eval import gtdt_match_info

    # This one is newer, but the results don't match up with pycocotools.
    from ._custom_eval import custom_eval

    # This one is based on pycocotools
    from ._pycocotools_eval import eval_with_pycocotools, eval_with_pycocotools_and_dump

from ._old_eval import GtDtMatchInfo, IdxMap