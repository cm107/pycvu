from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Results
    from .._structs import Annotations, Annotation
from pyevu import BBox2D
import numpy as np

@classmethod
def gtdt_match(cls, anns: Annotations, results: Results, thresh: float=0.1) -> dict[int, int | None]:
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

@classmethod
def gtdt_match_info(
    cls: type[Results],
    anns: Annotations, results: Results | Annotations, iouThresh: float=0.5,
    debug: bool=False
) -> GtDtMatchInfo:
    if type(results) is Annotations:
        results = cls.from_annotations(results)
    a2r = cls.gtdt_match(anns, results, thresh=iouThresh)
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
        print("Warning: GtDtMatchInfo is deprecated.")
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
        print("Warning: IdxMap is deprecated.")
        self.annIdx = annIdx
        self.rIdx = rIdx

    def __str__(self) -> str:
        paramStr = ','.join([f"{key}={val}" for key, val in self.__dict__.items()])
        return f"{type(self).__name__}({paramStr})"
    
    def __repr__(self) -> str:
        return self.__str__()