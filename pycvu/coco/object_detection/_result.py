from __future__ import annotations
from pyevu import BBox2D, Vector2
from ...polygon import Segmentation
from ...base import BaseHandler
from .._format import CocoBase
from ._structs import Annotation, Annotations

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
    
    @staticmethod
    def gtdt_match(anns: Annotations, results: Results) -> dict[int, int | None]:
        """
        Matches GT annotations to DT results by creating an index map.
        When a GT annotations has no matching DT result, it will be mapped to None.
        """
        _results = results.copy()
        _resultsIdxList = list(range(len(_results)))
        _resultsScoreList = [_r.score for _r in _results]
        _results = [_r for _, _r in sorted(zip(_resultsScoreList, _results), reverse=True)]
        _resultsIdxList = [_r for _, _r in sorted(zip(_resultsScoreList, _resultsIdxList), reverse=True)]
        _r2a: dict[int, int | None] = {}
        for i, r in enumerate(_results):
            ious: list[float] = []
            for j, ann in enumerate(anns):
                if ann.category_id == r.category_id and j not in _r2a.values():
                    iou = BBox2D.IoU(ann.bbox2d, r.bbox2d)
                else:
                    iou = 0
                ious.append(iou)
            maxIou = max(ious)
            maxIdx = ious.index(maxIou) if maxIou > 0 else None
            _r2a[i] = maxIdx
        a2r = {aIdx: _resultsIdxList[_rIdx] for _rIdx, aIdx in _r2a.items() if aIdx is not None}
        for k in range(len(anns)):
            if k not in a2r:
                a2r[k] = None
        assert None not in a2r.keys(), f"{a2r=}"
        return a2r
    
    @staticmethod
    def gtdt_match_info(anns: Annotations, results: Results, iouThresh: float=0.5) -> GtDtMatchInfo:
        a2r = Results.gtdt_match(anns, results)
        a2iou: dict[int, float] = {
            annIdx: (
                BBox2D.IoU(anns[annIdx].bbox2d, results[rIdx].bbox2d)
                if rIdx is not None
                else 0
            )
            for annIdx, rIdx in a2r.items()
        }
        tp: int = 0; fp: int = 0; fn: int = 0
        for annIdx, iou in a2iou.items():
            if iou >= iouThresh:
                tp += 1
            else:
                fp += 1; fn += 1
        for rIdx in range(len(results)):
            if rIdx not in a2r.values():
                fp += 1
        return GtDtMatchInfo(a2r=a2r, a2iou=a2iou, tp=tp, fp=fp, fn=fn)

class GtDtMatchInfo:
    def __init__(
        self,
        a2r: dict[int, int | None],
        a2iou: dict[int, float],
        tp: int, fp: int, fn: int
    ):
        self.a2r = a2r
        self.a2iou = a2iou
        self.tp = tp; self.fp = fp; self.fn = fn
    
    @property
    def tpfpfn(self) -> tuple[int, int, int]:
        return (self.tp, self.fp, self.fn)
