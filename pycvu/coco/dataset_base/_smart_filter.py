from __future__ import annotations
from typing import Callable, TypeVar
import random
import math
from datetime import datetime
import copy
from tqdm import tqdm
from .._format import CocoBase, CocoBaseHandler, Image, License, \
    ANN, CAT

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._base import DS

class SmartFilterMeta:
    def __init__(self):
        # Auxiliary
        self.licIdToLic: dict[int, tuple[int, License]] = {}
        self.imgIdToImg: dict[int, tuple[int, Image]] = {}
        self.annIdToAnn: dict[int, tuple[int, ANN]] = {}
        self.catIdToCat: dict[int, tuple[int, CAT]] = {}

        # Direct
        self.imgIdToAnns: dict[int, list[tuple[int, ANN]]] = {}
        self.catIdToAnns: dict[int, list[tuple[int, ANN]]] = {}
        self.licIdToImgs: dict[int, list[tuple[int, Image]]] = {}

        # Indirect
        self.imgIdToCats: dict[int, list[tuple[int, CAT]]] = {}
        self.catIdToImgs: dict[int, list[tuple[int, Image]]] = {}
        self.catIdToLics: dict[int, list[tuple[int, License]]] = {}
        self.licIdToAnns: dict[int, list[tuple[int, ANN]]] = {}
        self.licIdToCats: dict[int, list[tuple[int, CAT]]] = {}

        # Delete Queues
        self.licIdxDelQueue: list[int] = []
        self.imgIdxDelQueue: list[int] = []
        self.annIdxDelQueue: list[int] = []
        self.catIdxDelQueue: list[int] = []

    def init_maps(
        self, dataset: DS,
        # mapLics: bool=False,
        # mapImgs: bool=False,
        # mapAnns: bool=False,
        # mapCats: bool=False,
        showPbar: bool=False
    ):
        T = TypeVar('T')

        def update_aux_map(
            auxMap: dict[int, tuple[int, T]],
            objIdx: int, obj: T
        ):
            auxMap[obj.id] = (objIdx, obj)

        def update_map(
            map: dict[int, tuple[int, T]],
            key: int, objIdx: int, obj: T,
            checkExistence: bool=True
        ):
            if key in map:
                if (
                    not checkExistence
                    or objIdx not in [idx for idx, _ in map[key]]
                ):
                    map[key].append((objIdx, obj))
            else:
                map[key] = [(objIdx, obj)]

        # Direct & Auxiliary
        if True:
            lics = enumerate(dataset._licenses._objects)
            if showPbar:
                lics = tqdm(
                    lics, total=len(dataset._licenses),
                    desc="Forward mapping licenses"
                )
            for licIdx, lic in lics:
                lic: License = lic
                update_aux_map(self.licIdToLic, licIdx, lic)
        if True:
            imgs = enumerate(dataset._images._objects)
            if showPbar:
                imgs = tqdm(
                    imgs, total=len(dataset._images),
                    desc="Forward mapping images"
                )
            for imgIdx, img in imgs:
                img: Image = img
                update_aux_map(self.imgIdToImg, imgIdx, img)
                update_map(
                    self.licIdToImgs,
                    key=img.id, objIdx=imgIdx, obj=img,
                    checkExistence=False
                )
        if True:
            anns = enumerate(dataset._annotations._objects)
            if showPbar:
                anns = tqdm(
                    anns, total=len(dataset._annotations),
                    desc="Forward mapping annotations"
                )
            for annIdx, ann in anns:
                ann: ANN = ann
                update_aux_map(self.annIdToAnn, annIdx, ann)
                update_map(
                    self.imgIdToAnns,
                    key=ann.image_id, objIdx=annIdx, obj=ann,
                    checkExistence=False
                )
                update_map(
                    self.catIdToAnns,
                    key=ann.category_id, objIdx=annIdx, obj=ann,
                    checkExistence=False
                )
        if True:
            cats = enumerate(dataset._categories._objects)
            if showPbar:
                cats = tqdm(
                    cats, total=len(dataset._categories),
                    desc="Forward mapping categories"
                )
            for catIdx, cat in cats:
                cat: CAT = cat
                update_aux_map(self.catIdToCat, catIdx, cat)

        # Indirect
        if True:
            catPairs = self.catIdToCat.items()
            if showPbar:
                catPairs = tqdm(
                    catPairs, total=len(catPairs),
                    desc="Reverse mapping categories"
                )
            for catId, (catIdx, cat) in catPairs:
                catAnns = self.catIdToAnns[catId]
                if showPbar:
                    catAnns = tqdm(catAnns, desc="Reverse mapping Category Annotations", leave=False)
                for annIdx, ann in catAnns:
                    imgId = ann.image_id
                    imgIdx, img = self.imgIdToImg[imgId]
                    licId = img.license
                    licIdx, lic = self.licIdToLic[licId]
                    
                    update_map(
                        self.imgIdToCats,
                        key=imgId, objIdx=catIdx, obj=cat
                    )
                    update_map(
                        self.catIdToImgs,
                        key=catId, objIdx=imgIdx, obj=img
                    )
                    update_map(
                        self.catIdToLics,
                        key=catId, objIdx=licIdx, obj=lic
                    )
                    update_map(
                        self.licIdToAnns,
                        key=licId, objIdx=annIdx, obj=ann
                    )
                    update_map(
                        self.licIdToCats,
                        key=licId, objIdx=catIdx, obj=cat
                    )

    def cleanup(self, dataset: DS, showPbar: bool=False):
        # Finish deciding which indicies to delete.
        affectedLicenseIdxList: list[int] = []
        affectedImgIdxList: list[int] = []
        # affectedAnnIdxList: list[int] = []
        affectedCatIdxList: list[int] = []

        
        if len(self.catIdxDelQueue) > 0:
            _queue = self.catIdxDelQueue
            if showPbar:
                _queue = tqdm(_queue, desc='Checking Affects Of Deleting Categories')
            for catIdx in _queue:
                cat: CAT = dataset._categories[catIdx]
                for annIdx, ann in self.catIdToAnns[cat.id]:
                    if annIdx not in self.annIdxDelQueue:
                        self.annIdxDelQueue.append(annIdx)
                    imgIdx, img = self.imgIdToImg[ann.image_id]
                    if imgIdx not in affectedImgIdxList:
                        affectedImgIdxList.append(imgIdx)
        if len(self.annIdxDelQueue) > 0:
            _queue = self.annIdxDelQueue
            if showPbar:
                _queue = tqdm(_queue, desc='Checking Affects Of Deleting Annotations')
            for annIdx in _queue:
                ann: ANN = dataset._annotations[annIdx]
                imgIdx, img = self.imgIdToImg[ann.image_id]
                if imgIdx not in affectedImgIdxList:
                    affectedImgIdxList.append(imgIdx)
                catIdx, cat = self.catIdToCat[ann.category_id]
                if catIdx not in affectedCatIdxList:
                    affectedCatIdxList.append(catIdx)
        if len(self.imgIdxDelQueue) > 0:
            _queue = self.imgIdxDelQueue
            if showPbar:
                _queue = tqdm(_queue, desc='Checking Affects Of Deleting Images')
            for imgIdx in _queue:
                img: Image = dataset._images[imgIdx]
                for annIdx, ann in self.imgIdToAnns[img.id]:
                    if annIdx not in self.annIdxDelQueue:
                        self.annIdxDelQueue.append(annIdx)
                    catIdx, cat = self.catIdToCat[ann.category_id]
                    if catIdx not in affectedCatIdxList:
                        affectedCatIdxList.append(catIdx)
                licIdx, lic = self.licIdToLic[img.license]
                if licIdx not in affectedLicenseIdxList:
                    affectedLicenseIdxList.append(licIdx)
        if len(self.licIdxDelQueue) > 0:
            _queue = self.licIdxDelQueue
            if showPbar:
                _queue = tqdm(_queue, desc='Checking Affects Of Deleting Licenses')
            for licIdx in _queue:
                lic: License = dataset._licenses[licIdx]
                for imgIdx, img in self.licIdToImgs[lic.id]:
                    if imgIdx not in self.imgIdxDelQueue:
                        self.imgIdxDelQueue.append(imgIdx)
                    for annIdx, ann in self.imgIdToAnns[img.id]:
                        if annIdx not in self.annIdxDelQueue:
                            self.annIdxDelQueue.append(annIdx)
                        catIdx, cat = self.catIdToCat[ann.category_id]
                        if catIdx not in affectedCatIdxList:
                            affectedCatIdxList.append(catIdx)
        
        affectedCatIdxList = list(set(affectedCatIdxList) - set(self.catIdxDelQueue))
        if len(affectedCatIdxList) > 0:
            _affected = affectedCatIdxList
            if showPbar:
                _affected = tqdm(_affected, desc='Processing Affected Categories')
            for catIdx in _affected:
                cat = dataset._categories[catIdx]
                stillBeingUsed = False
                for annIdx, ann in self.catIdToAnns[cat.id]:
                    if annIdx not in self.annIdxDelQueue:
                        stillBeingUsed = True
                        break
                if not stillBeingUsed:
                    self.catIdxDelQueue.append(catIdx)
        affectedImgIdxList = list(set(affectedImgIdxList) - set(self.imgIdxDelQueue))
        if len(affectedImgIdxList) > 0:
            _affected = affectedImgIdxList
            if showPbar:
                _affected = tqdm(_affected, 'Processing Affected Images')
            for imgIdx in _affected:
                img = dataset._images[imgIdx]
                stillBeingUsed = False
                for annIdx, ann in self.imgIdToAnns[img.id]:
                    if annIdx not in self.annIdxDelQueue:
                        stillBeingUsed = True
                        break
                if not stillBeingUsed:
                    self.imgIdxDelQueue.append(imgIdx)
                    licIdx, lic in self.licIdToLic[img.license]
                    if licIdx not in affectedLicenseIdxList:
                        affectedLicenseIdxList.append(licIdx)
        affectedLicenseIdxList = list(set(affectedLicenseIdxList) - set(self.licIdxDelQueue))
        if len(affectedLicenseIdxList) > 0:
            _affected = affectedLicenseIdxList
            if showPbar:
                _affected = tqdm(_affected, 'Processing Affected Licenses')
            for licIdx in _affected:
                lic = dataset._licenses[licIdx]
                stillBeingUsed = False
                for imgIdx, img in self.licIdToImgs[lic.id]:
                    if imgIdx not in self.imgIdxDelQueue:
                        stillBeingUsed = True
                        break
                if not stillBeingUsed:
                    self.licIdxDelQueue.append(licIdx)
        
        # Delete everything in the queue
        _delQueue = sorted(self.catIdxDelQueue)[::-1]
        if len(_delQueue) > 0:
            if showPbar:
                _delQueue = tqdm(_delQueue, desc='Deleting Categories')
            for idx in _delQueue:
                del dataset._categories[idx]
        _delQueue = sorted(self.annIdxDelQueue)[::-1]
        if len(_delQueue) > 0:
            if showPbar:
                _delQueue = tqdm(_delQueue, 'Deleting Annotations')
            for idx in _delQueue:
                del dataset._annotations[idx]
        _delQueue = sorted(self.imgIdxDelQueue)[::-1]
        if len(_delQueue) > 0:
            if showPbar:
                _delQueue = tqdm(_delQueue, 'Deleting Images')
            for idx in _delQueue:
                del dataset._images[idx]
        _delQueue = sorted(self.licIdxDelQueue)[::-1]
        if len(_delQueue) > 0:
            if showPbar:
                _delQueue = tqdm(_delQueue, 'Deleting Licenses')
            for idx in _delQueue:
                del dataset._licenses[idx]

def smart_filter(
    self: DS,
    catFilter: Callable[[CAT], bool]=None,
    annFilter: Callable[[ANN], bool]=None,
    imgFilter: Callable[[Image], bool]=None,
    licFilter: Callable[[License], bool]=None,
    # cleanup: bool | None=None,
    # cleanupCat: bool=True, cleanupAnn: bool=True, cleanupImg: bool=True, cleanupLic: bool=True,
    reindex: bool=False,
    showPbar: bool=False, leavePbar: bool=False,
    applyToSelf: bool=False
) -> DS:
    """
    Purpose:
        When filtering a large dataset with a lot of images and a lot of annotations, smart_filter MAY be faster than filter. (Still looking for a use case.)
        However, when working with small datasets that have few images or few annotations, filter can be faster than smart_filter.
    Disclaimer:
        This method has not yet been adequately tested.
        Therefore, it may yield undesirable results.
        Use it at your own discretion.
        If possible, you may want to use the filter method instead to be safe.
    """
    print("Disclaimer: smart_filter has not been adequately tested yet. May yield undesirable results.")
    # if cleanup is not None:
    #     cleanupCat = cleanup
    #     cleanupAnn = cleanup
    #     cleanupImg = cleanup
    #     cleanupLic = cleanup
    if catFilter is None and annFilter is None and imgFilter is None and licFilter is None:
        raise ValueError("Must specify at least one filter callback.")
    
    # Apply Filters (Easy Part)
    if not applyToSelf:
        if showPbar:
            pbar = tqdm(total=1, leave=leavePbar)
            pbar.set_description("Copying Data For Filtering")
        result = copy.deepcopy(self)
        if showPbar:
            pbar.update()
            pbar.close()
    else:
        result = self
    
    meta = SmartFilterMeta()
    meta.init_maps(result, showPbar=showPbar)

    for handler, callback, delQueue in [
        (result._categories, catFilter, meta.catIdxDelQueue),
        (result._annotations, annFilter, meta.annIdxDelQueue),
        (result._images, imgFilter, meta.imgIdxDelQueue),
        (result._licenses, licFilter, meta.licIdxDelQueue)
    ]:
        if callback is None:
            continue
        handler: CocoBaseHandler[CocoBase] = handler
        callback: Callable[[CocoBase], bool] = callback
        delQueue: list[int] = delQueue
        if callback is not None:
            if showPbar:
                pbar = tqdm(total=len(handler), leave=leavePbar)
                pbar.set_description(f"Filtering {type(handler).__name__}")
            for idx in list(range(len(handler))):
                obj = handler[idx]
                if not callback(obj):
                    delQueue.append(idx)
                if showPbar:
                    pbar.update()
            if showPbar:
                pbar.close()
    
    meta.cleanup(result, showPbar=showPbar)
    if reindex:
        self.reindex(applyToSelf=applyToSelf, showPbar=showPbar, leavePbar=leavePbar)
