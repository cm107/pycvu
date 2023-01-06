from __future__ import annotations
from pyevu import BBox2D
from .._result import Result, Results
from .._structs import Annotation, Annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Dataset

class Pair:
    def __init__(self, ann: Annotation, r: Result):
        self.ann = ann
        self.r = r

    @property
    def iou(self) -> float:
        if self.r.score is None:
            return 0
        else:
            return BBox2D.IoU(self.ann.bbox2d, self.r.bbox2d)

class IouMeta:
    iouThresh: float = 0.5

    def __init__(
        self, ious: list[float], resultIdxList: list[int]
    ):
        self._ious = ious
        self._resultIdxList = resultIdxList
        self._k = 0
    
    @property
    def resultIdx(self) -> int | None:
        if self._k >= len(self._resultIdxList) \
            or self._ious[self._resultIdxList[self._k]] < IouMeta.iouThresh:
            return None
        else:
            return self._resultIdxList[self._k]
    
    @property
    def iou(self) -> float | None:
        idx = self.resultIdx
        if idx is not None:
            return self._ious[idx]
        else:
            return None

    def next(self):
        self._k += 1

class FrameEvalMeta:
    def __init__(
        self, anns: Annotations, results: Results,
        tpResultIdxList: list[int], fpResultIdxList: list[int],
        fnAnnIdxList: list[int], tpAnnIdxList: list[int]
    ):
        self.anns = anns; self.results = results
        self.tpResultIdxList = tpResultIdxList
        self.fpResultIdxList = fpResultIdxList
        self.fnAnnIdxList = fnAnnIdxList
        self.tpAnnIdxList = tpAnnIdxList
    
    @property
    def tpResults(self) -> Results:
        return Results(
            [
                r for idx, r in enumerate(self.results)
                if idx in self.tpResultIdxList
            ]
        )
    
    @property
    def fpResults(self) -> Results:
        return Results(
            [
                r for idx, r in enumerate(self.results)
                if idx in self.fpResultIdxList
            ]
        )

    @property
    def tpAnns(self) -> Annotations:
        return Annotations(
            [
                ann for idx, ann in enumerate(self.anns)
                if idx in self.tpAnnIdxList
            ]
        )

    @property
    def fnAnns(self) -> Annotations:
        return Annotations(
            [
                ann for idx, ann in enumerate(self.anns)
                if idx in self.fnAnnIdxList
            ]
        )

    @classmethod
    def from_data(
        cls,
        anns: Annotations, results: Results,
        iouThresh: float=0.5
    ) -> FrameEvalMeta:
        IouMeta.iouThresh = iouThresh
        iouMetaList: list[IouMeta] = []
        for i, ann in enumerate(anns):
            pairs = [Pair(ann, r) for r in results]
            ious = [pair.iou for pair in pairs]

            idxList: list[int] = list(range(len(ious)))
            resultIdxList = [idx for _, idx in sorted(list(zip(ious, idxList)), reverse=True)]
            
            iouMeta = IouMeta(ious, resultIdxList)
            iouMetaList.append(iouMeta)
        
        matchedResultIdxList: list[int] = []
        for i in list(range(len(iouMetaList)))[::-1]:
            iouMeta = iouMetaList[i]
            while iouMeta.resultIdx in matchedResultIdxList:
                iouMeta.next()

            while iouMeta.resultIdx is not None:
                redoFlag: bool = False
                for j in range(0, i):
                    otherIouMeta = iouMetaList[j]
                    if otherIouMeta.resultIdx is None:
                        continue
                    elif iouMeta.resultIdx == otherIouMeta.resultIdx:
                        if iouMeta.iou < otherIouMeta.iou:
                            iouMeta.next()
                            redoFlag = True
                            break
                if not redoFlag:
                    break
            
            if iouMeta.resultIdx is not None:
                matchedResultIdxList.append(iouMeta.resultIdx)

        tpResultIdxList: list[int] = [
            iouMeta.resultIdx for iouMeta in iouMetaList
            if iouMeta.resultIdx is not None
        ]
        fpResultIdxList: list[int] = [
            idx for idx in range(len(results))
            if idx not in tpResultIdxList and results[idx].score is not None
        ]
        fnAnnIdxList: list[int] = [
            idx for idx, iouMeta in enumerate(iouMetaList)
            if iouMeta.resultIdx is None
        ]
        tpAnnIdxList: list[int] = [
            idx for idx, iouMeta in enumerate(iouMetaList)
            if iouMeta.resultIdx is not None
        ]
        return cls(
            anns=anns, results=results,
            tpResultIdxList=tpResultIdxList,
            fpResultIdxList=fpResultIdxList,
            fnAnnIdxList=fnAnnIdxList,
            tpAnnIdxList=tpAnnIdxList
        )

def get_frame_eval_meta(
    self: Dataset, dt: Results, image_id: int,
    iouThresh: float=0.5
) -> FrameEvalMeta:
    anns = self.annotations.search(lambda ann: ann.image_id == image_id)
    results = dt.search(lambda r: r.image_id == image_id)
    return FrameEvalMeta.from_data(
        anns=anns, results=results,
        iouThresh=iouThresh
    )
