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
    
    # @staticmethod
    # def gtdt_match(anns: Annotations, results: Results, debug: bool=False) -> dict[int, int | None]:
    #     """
    #     Matches GT annotations to DT results by creating an index map.
    #     When a GT annotations has no matching DT result, it will be mapped to None.
    #     """
    #     _results = results.copy()
    #     _resultsIdxList = list(range(len(_results)))
    #     _resultsScoreList = [_r.score for _r in _results]
    #     _results = [_r for _, _r in sorted(zip(_resultsScoreList, _results), reverse=True)]
    #     _resultsIdxList = [_r for _, _r in sorted(zip(_resultsScoreList, _resultsIdxList), reverse=True)]
    #     _r2a: dict[int, int | None] = {}
    #     _iousList: list[list[float]] = []
    #     for i, r in enumerate(_results):
    #         ious: list[float] = []
    #         if len(anns) > 0:
    #             for j, ann in enumerate(anns):
    #                 if j not in _r2a.values():
    #                     if ann.category_id == r.category_id:
    #                         iou = BBox2D.IoU(ann.bbox2d, r.bbox2d)
    #                     else:
    #                         iou = 0
    #                     ious.append(iou)
    #                 else:
    #                     if ann.category_id == r.category_id:
    #                         iou = BBox2D.IoU(ann.bbox2d, r.bbox2d)
    #                         # _iou: float = None
    #                         # for _rIdx, _aIdx in _r2a.items():
    #                         #     if j == _aIdx:
    #                         #         _iou = BBox2D.IoU(_results[_rIdx].bbox2d, anns[_aIdx].bbox2d)
    #                         #         break
    #                         # assert _iou is not None
    #                         _annIdx = j
    #                         _rIdx = None
    #                         for _rIdx0, _aIdx0 in _r2a.items():
    #                             if _aIdx0 == _annIdx:
    #                                 _rIdx = _rIdx0
    #                                 break
    #                         assert _rIdx is not None
    #                         _ious = _iousList[_rIdx]
    #                         _iou = _ious[_annIdx]
    #                         if iou > _iou:
    #                             _ious[_annIdx] = 0
    #                             _maxIou = max(_ious)
    #                             _maxIdx = _ious.index(_maxIou) if _maxIou > 0 else None
    #                             _iousList[_rIdx] = _ious
    #                             _r2a[_rIdx] = _maxIdx
    #                             ious.append(iou)

    #             maxIou = max(ious)
    #             maxIdx = ious.index(maxIou) if maxIou > 0 else None
    #             if debug:
    #                 print(f"\t{i} {ious=}")
    #             _r2a[i] = maxIdx
    #         else:
    #             _r2a[i] = None
    #         _iousList.append(ious)
    #     a2r = {aIdx: _resultsIdxList[_rIdx] for _rIdx, aIdx in _r2a.items() if aIdx is not None}
    #     for k in range(len(anns)):
    #         if k not in a2r:
    #             a2r[k] = None
    #     assert None not in a2r.keys(), f"{a2r=}"
    #     return a2r

    @staticmethod
    def gtdt_match(anns: Annotations, results: Results, thresh: float=0.1) -> dict[int, int | None]:
        import numpy as np
        iouMat = np.zeros((anns.__len__(), results.__len__()))
        for i in range(len(anns)):
            ann: Annotation = anns[i]
            for j in range(len(results)):
                result = results[j]
                gtBBox = ann.bbox2d
                dtBBox = result.bbox2d
                if ann.category_id == result.category_id:
                    iou = BBox2D.IoU(gtBBox, dtBBox)
                else:
                    iou = 0
                iouMat[i, j] = iou
        # print(iouMat)
        # print(f"{iouMat.shape=}")
        gt2dtMatchIdxList = iouMat.argmax(axis=1).tolist() if iouMat.shape[1] > 0 else [None] * iouMat.shape[0]
        gt2dtMatchIouList = [iouMat[gtIdx, dtIdx] for gtIdx, dtIdx in enumerate(gt2dtMatchIdxList)]
        for i in list(range(len(gt2dtMatchIdxList)))[::-1]:
            if gt2dtMatchIouList[i] < thresh:
                gt2dtMatchIdxList[i] = None
                gt2dtMatchIouList[i] = 0
                # del gt2dtMatchIdxList[i]
                # del gt2dtMatchIouList[i]
        # print(f"{gt2dtMatchIdxList=}")
        # print(f"{gt2dtMatchIouList=}")

        a2rIdxMatch = {annIdx: rIdx for annIdx, rIdx in enumerate(gt2dtMatchIdxList)}
        return a2rIdxMatch
    
    @staticmethod
    def gtdt_match0(anns: Annotations, results: Results, debug: bool=False) -> dict[int, int | None]:
        """
        Matches GT annotations to DT results by creating an index map.
        When a GT annotations has no matching DT result, it will be mapped to None.
        """
        # TODO: Debug
        _results = results.copy()
        _resultsIdxList = list(range(len(_results)))
        _resultsScoreList = [_r.score for _r in _results]
        _results = [_r for _, _r in sorted(zip(_resultsScoreList, _results), reverse=True)]
        _resultsIdxList = [_r for _, _r in sorted(zip(_resultsScoreList, _resultsIdxList), reverse=True)]
        _r2a: dict[int, int | None] = {}
        for i, r in enumerate(_results):
            if len(anns) > 0:
                ious: list[float] = []
                for j, ann in enumerate(anns):
                    if ann.category_id == r.category_id and j not in _r2a.values(): # Problem is here.
                        iou = BBox2D.IoU(ann.bbox2d, r.bbox2d)
                    else:
                        iou = 0
                    ious.append(iou)
                maxIou = max(ious)
                maxIdx = ious.index(maxIou) if maxIou > 0 else None
                _r2a[i] = maxIdx
            else:
                _r2a[i] = None
        a2r = {aIdx: _resultsIdxList[_rIdx] for _rIdx, aIdx in _r2a.items() if aIdx is not None}
        for k in range(len(anns)):
            if k not in a2r:
                a2r[k] = None
        assert None not in a2r.keys(), f"{a2r=}"
        return a2r
    
    @staticmethod
    def gtdt_match_info0(
        anns: Annotations, results: Results, iouThresh: float=0.5,
        debug: bool=False
    ) -> GtDtMatchInfo:
        a2r = Results.gtdt_match0(anns, results)
        a2iou: dict[int, float] = {
            annIdx: (
                BBox2D.IoU(anns[annIdx].bbox2d, results[rIdx].bbox2d)
                if rIdx is not None
                else 0
            )
            for annIdx, rIdx in a2r.items()
        }
        # tp: int = 0; fp: int = 0; fn: int = 0
        tpIdxMaps: list[IdxMap] = []
        fpIdxMaps: list[IdxMap] = []
        fnIdxMaps: list[IdxMap] = []
        for annIdx, iou in a2iou.items():
            rIdx: int | None = a2r[annIdx] if annIdx in a2r else None
            if debug:
                print(f"\t{annIdx=}, {iou=}")
                print(f"\t{rIdx=}")
            if rIdx is not None:
                if iou >= iouThresh:
                    # tp += 1
                    tpIdxMaps.append(IdxMap(annIdx=annIdx, rIdx=rIdx))
                else:
                    # fp += 1; fn += 1
                    fpIdxMaps.append(IdxMap(annIdx=None, rIdx=rIdx))
                    fnIdxMaps.append(IdxMap(annIdx=annIdx, rIdx=None))
            else:
                # fn += 1
                fnIdxMaps.append(IdxMap(annIdx=None, rIdx=rIdx))
        for rIdx in range(len(results)):
            annIdx: int | None = None
            if rIdx not in a2r.values():
                # fp += 1
                fpIdxMaps.append(IdxMap(annIdx=None, rIdx=rIdx))
        return GtDtMatchInfo(
            a2r=a2r, a2iou=a2iou,
            tpIdxMaps=tpIdxMaps,
            fpIdxMaps=fpIdxMaps,
            fnIdxMaps=fnIdxMaps
        )

    @staticmethod
    def gtdt_match_info(
        anns: Annotations, results: Results | Annotations, iouThresh: float=0.5,
        debug: bool=False
    ) -> GtDtMatchInfo:
        if type(results) is Annotations:
            results = Results.from_annotations(results)
        a2r = Results.gtdt_match(anns, results, thresh=iouThresh)
        a2iou: dict[int, float] = {
            annIdx: (
                BBox2D.IoU(anns[annIdx].bbox2d, results[rIdx].bbox2d)
                if rIdx is not None
                else 0
            )
            for annIdx, rIdx in a2r.items()
        }
        # tp: int = 0; fp: int = 0; fn: int = 0
        tpIdxMaps: list[IdxMap] = []
        fpIdxMaps: list[IdxMap] = []
        fnIdxMaps: list[IdxMap] = []
        for annIdx, iou in a2iou.items():
            rIdx: int | None = a2r[annIdx] if annIdx in a2r else None
            if debug:
                print(f"\t{annIdx=}, {iou=}")
                print(f"\t{rIdx=}")
            if rIdx is not None:
                if iou >= iouThresh:
                    # tp += 1
                    tpIdxMaps.append(IdxMap(annIdx=annIdx, rIdx=rIdx))
                else:
                    # fp += 1; fn += 1
                    fpIdxMaps.append(IdxMap(annIdx=None, rIdx=rIdx))
                    fnIdxMaps.append(IdxMap(annIdx=annIdx, rIdx=None))
            else:
                # fn += 1
                fnIdxMaps.append(IdxMap(annIdx=None, rIdx=rIdx))
        for rIdx in range(len(results)):
            annIdx: int | None = None
            if rIdx not in a2r.values():
                # fp += 1
                fpIdxMaps.append(IdxMap(annIdx=None, rIdx=rIdx))
        return GtDtMatchInfo(
            a2r=a2r, a2iou=a2iou,
            tpIdxMaps=tpIdxMaps,
            fpIdxMaps=fpIdxMaps,
            fnIdxMaps=fnIdxMaps
        )

    @staticmethod
    def debug_gtdt_match():
        from pycvu.coco.object_detection import Dataset
        ds = Dataset.from_labelme(annDir="/home/clayton/Pictures/bbox_match_test/json")
        gtBboxList: list[BBox2D] = []
        dtBboxList: list[BBox2D] = []
        for ann in ds.annotations:
            cat = ds.categories.get(lambda cat: cat.id == ann.category_id)
            if cat.name == 'dt':
                dtBboxList.append(ann.bbox2d)
            elif cat.name == 'gt':
                gtBboxList.append(ann.bbox2d)
            else:
                raise Exception
        
        print(f"{len(gtBboxList)=}")
        print(f"{len(dtBboxList)=}")
        anns = Annotations()
        results = Results()
        for bbox in gtBboxList:
            ann = Annotation(
                id=len(anns), image_id=0, category_id=0, segmentation=[], area=bbox.area,
                bbox=[bbox.v0.x, bbox.v0.y, bbox.xInterval.length, bbox.yInterval.length],
                iscrowd=0
            )
            anns.append(ann)
        for bbox in dtBboxList:
            result = BBoxResult(
                image_id=0, category_id=0,
                bbox=[bbox.v0.x, bbox.v0.y, bbox.xInterval.length, bbox.yInterval.length],
                score=0.8
            )
            results.append(result)
        
        Results.gtdt_match(anns=anns, results=results)

    @staticmethod
    def debug_gtdt_match_info():
        from pycvu.coco.object_detection import Dataset
        ds = Dataset.from_labelme(annDir="/home/clayton/Pictures/bbox_match_test/json")
        gtBboxList: list[BBox2D] = []
        dtBboxList: list[BBox2D] = []
        for ann in ds.annotations:
            cat = ds.categories.get(lambda cat: cat.id == ann.category_id)
            if cat.name == 'dt':
                dtBboxList.append(ann.bbox2d)
            elif cat.name == 'gt':
                gtBboxList.append(ann.bbox2d)
            else:
                raise Exception
        
        print(f"{len(gtBboxList)=}")
        print(f"{len(dtBboxList)=}")
        anns = Annotations()
        results = Results()
        for bbox in gtBboxList:
            ann = Annotation(
                id=len(anns), image_id=0, category_id=0, segmentation=[], area=bbox.area,
                bbox=[bbox.v0.x, bbox.v0.y, bbox.xInterval.length, bbox.yInterval.length],
                iscrowd=0
            )
            anns.append(ann)
        for bbox in dtBboxList:
            result = BBoxResult(
                image_id=0, category_id=0,
                bbox=[bbox.v0.x, bbox.v0.y, bbox.xInterval.length, bbox.yInterval.length],
                score=0.8
            )
            results.append(result)
        
        info = Results.gtdt_match_info(anns=anns, results=results, iouThresh=0.1)
        print(info.tpfpfn)

class GtDtMatchInfo:
    def __init__(
        self,
        a2r: dict[int, int | None],
        a2iou: dict[int, float],
        # tp: int, fp: int, fn: int
        tpIdxMaps: list[IdxMap],
        fpIdxMaps: list[IdxMap],
        fnIdxMaps: list[IdxMap]
    ):
        self.a2r = a2r
        self.a2iou = a2iou
        # self.tp = tp; self.fp = fp; self.fn = fn
        self.tpIdxMaps = tpIdxMaps
        self.fpIdxMaps = fpIdxMaps
        self.fnIdxMaps = fnIdxMaps
    
    @property
    def tp(self) -> int:
        return len(self.tpIdxMaps)
    
    @property
    def fp(self) -> int:
        return len(self.fpIdxMaps)
    
    @property
    def fn(self) -> int:
        return len(self.fnIdxMaps)
    
    @property
    def tpfpfn(self) -> tuple[int, int, int]:
        return (self.tp, self.fp, self.fn)

class IdxMap:
    def __init__(self, annIdx: int | None, rIdx: int | None):
        self.annIdx = annIdx
        self.rIdx = rIdx

    def __str__(self) -> str:
        paramStr = ','.join([f"{key}={val}" for key, val in self.__dict__.items()])
        return f"{type(self).__name__}({paramStr})"
    
    def __repr__(self) -> str:
        return self.__str__()