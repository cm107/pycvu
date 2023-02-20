from __future__ import annotations
from typing import Callable
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

@classmethod
def combine(cls: type[DS], sources: list[DS | str], showPbar: bool=False) -> DS:
    combined = cls()

    # Process Info
    combined._info.description = "Combined dataset using pycvu."
    combined._info.date_created = datetime.now()
    combined._info.year = datetime.now().year

    outerPbar = tqdm(total=len(sources), leave=True) if showPbar else None
    if showPbar:
        outerPbar.set_description("Combining Datasets")
    for src in sources:
        if type(src) is str:
            dataset = cls.load(src)
        elif type(src) is cls:
            dataset = src
        else:
            raise TypeError
        
        innerPbar = tqdm(total=len(dataset._images), leave=False) if showPbar else None
        dataset._images.sort(lambda image: image.id)
        for image in dataset._images:
            # Process License
            license = dataset._licenses.get(lambda lic: lic.id == image.license)
            assert license is not None
            lic = combined._licenses.get(
                lambda lic: all([
                    getattr(lic, key) == getattr(license, key)
                    for key in license.__dict__
                    if key != 'id'
                ])
            )
            if lic is None:
                # New license.
                lic = license.copy()
                lic.id = len(combined._licenses)
                combined._licenses.append(lic)

            # Process Image
            img = image.copy()
            img.id = len(combined._images)
            img.license = lic.id
            combined._images.append(img)

            for annotation in dataset._annotations.search(lambda ann: ann.image_id == image.id):
                # Process Category
                category = dataset._categories.get(lambda cat: cat.id == annotation.category_id)
                assert category is not None
                cat = combined._categories.get(
                    lambda cat: all([
                        getattr(cat, key) == getattr(category, key)
                        for key in category.__dict__
                        if key != 'id'
                    ])
                )
                if cat is None:
                    # New category.
                    cat = category.copy()
                    cat.id = len(combined._categories)
                    combined._categories.append(cat)

                # Process Annotation
                ann = annotation.copy()
                ann.id = len(combined._annotations)
                ann.image_id = img.id
                ann.category_id = cat.id
                combined._annotations.append(ann)
            if showPbar:
                innerPbar.update()
        if showPbar:
            innerPbar.close()
            outerPbar.update()
    if showPbar:
        outerPbar.close()
    return combined

def filter(
    self: DS,
    catFilter: Callable[[CAT], bool]=None,
    annFilter: Callable[[ANN], bool]=None,
    imgFilter: Callable[[Image], bool]=None,
    licFilter: Callable[[License], bool]=None,
    cleanup: bool | None=None,
    cleanupCat: bool=True, cleanupAnn: bool=True, cleanupImg: bool=True, cleanupLic: bool=True,
    reindex: bool=False,
    showPbar: bool=False, leavePbar: bool=False,
    applyToSelf: bool=False
) -> DS:
    # TODO: docstring
    # Note: Each filter affects all handlers either directly or through broken references.
    # Example: catFilter affects categories directly, and affects annotations, images, and
    #          licenses through broken references.
    if cleanup is not None:
        cleanupCat = cleanup
        cleanupAnn = cleanup
        cleanupImg = cleanup
        cleanupLic = cleanup
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
    for handler, callback in [
        (result._categories, catFilter),
        (result._annotations, annFilter),
        (result._images, imgFilter),
        (result._licenses, licFilter)
    ]:
        if callback is None:
            continue
        handler: CocoBaseHandler[CocoBase] = handler
        callback: Callable[[CocoBase], bool] = callback
        if callback is not None:
            if showPbar:
                pbar = tqdm(total=len(handler), leave=leavePbar)
                pbar.set_description(f"Filtering {type(handler).__name__}")
            for idx in list(range(len(handler)))[::-1]:
                obj = handler[idx]
                if not callback(obj):
                    del handler[idx]
                if showPbar:
                    pbar.update()
            if showPbar:
                pbar.close()

    # Cleanup Broken References (Hard Part)
    if catFilter is not None and cleanupAnn:
        # Categories affect Annotations
        if showPbar:
            pbar = tqdm(total=len(result._annotations), leave=leavePbar)
            pbar.set_description("Cleanup Annotations w/o Categories")
        for idx in list(range(len(result._annotations)))[::-1]:
            ann = result._annotations[idx]
            if result._categories.get(id=ann.category_id) is None:
                del result._annotations[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if licFilter is not None and cleanupImg:
        # Licenses affect Images
        if showPbar:
            pbar = tqdm(total=len(result._images), leave=leavePbar)
            pbar.set_description("Cleanup Images w/o Licenses")
        for idx in list(range(len(result._images)))[::-1]:
            img = result._images[idx]
            if result._licenses.get(id=img.license) is None:
                del result._images[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if (annFilter is not None or catFilter is not None) and cleanupImg:
        # Annotations affect Images -> Images Done
        if showPbar:
            pbar = tqdm(total=len(result._images), leave=leavePbar)
            pbar.set_description("Cleanup Images w/o Annotations")
        for idx in list(range(len(result._images)))[::-1]:
            img = result._images[idx]
            if result._annotations.get(image_id=img.id) is None:
                del result._images[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if (imgFilter is not None or annFilter is not None or catFilter is not None) and cleanupLic:
        # Images affect Licenses -> Licenses Done
        if showPbar:
            pbar = tqdm(total=len(result._licenses), leave=leavePbar)
            pbar.set_description("Cleanup Licenses w/o Images")
        for idx in list(range(len(result._licenses)))[::-1]:
            lic = result._licenses[idx]
            if result._images.get(license=lic.id) is None:
                del result._licenses[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if (imgFilter is not None or licFilter is not None) and cleanupAnn:
        # Images affect Annotations -> Annotations Done
        if showPbar:
            pbar = tqdm(total=len(result._annotations), leave=leavePbar)
            pbar.set_description("Cleanup Annotations w/o Images")
        for idx in list(range(len(result._annotations)))[::-1]:
            ann = result._annotations[idx]
            if result._images.get(id=ann.image_id) is None:
                del result._annotations[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if (annFilter is not None or imgFilter is not None or licFilter is not None) and cleanupCat:
        # Annotations affect Categories -> Categories Done
        if showPbar:
            pbar = tqdm(total=len(result._categories), leave=leavePbar)
            pbar.set_description("Cleanup Categories w/o Annotations")
        for idx in list(range(len(result._categories)))[::-1]:
            cat = result._categories[idx]
            if result._annotations.get(category_id=cat.id) is None:
                del result._categories[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()

    if reindex:
        result.reindex(applyToSelf=True, showPbar=showPbar, leavePbar=leavePbar)

    return result

def reindex(
    self: DS, applyToSelf: bool=True,
    showPbar: bool=False, leavePbar: bool=False
) -> DS:
    if not applyToSelf:
        result = copy.deepcopy(self)
    else:
        result = self
    
    def updateLicenseIdInImages(oldId: int, newId: int):
        for img in result._images.search(lambda img: img.license == oldId):
            img.license = newId

    def updateImgIdInAnnotations(oldId: int, newId: int):
        for ann in result._annotations.search(lambda ann: ann.image_id == oldId):
            ann.image_id = newId
    
    def updateCatIdInAnnotations(oldId: int, newId: int):
        for ann in result._annotations.search(lambda ann: ann.category_id == oldId):
            ann.category_id = newId

    for handler, idUpdateCallback in [
        (result._licenses, updateLicenseIdInImages),
        (result._images, updateImgIdInAnnotations),
        (result._categories, updateCatIdInAnnotations),
        (result._annotations, None)
    ]:
        handler: CocoBaseHandler = handler
        handler.reindex(
            showPbar=showPbar, leavePbar=leavePbar,
            applyToSelf=True,
            idUpdateCallback=idUpdateCallback
        )
    
    return result

def split(
    self: DS,
    weights: list[1, 1, 1],
    seed: None | int | float | str | bytes=None,
    showPbar: bool=False, leavePbar: bool=False
) -> list[DS]:
    """
    Splits the dataset up into parts.

    Parameters:
        seed:
            Seed used for random number generator when randomly
            splitting the dataset.
            Note:
                If left as None, the system time is used, meaning
                the results won't be reproducible.
        showPbar:
            Show the progress bar.
        leavePbar:
            Leave the progress bar.
    """
    rand = random.Random(seed)
    sampleProps: float = [weights[i]/(sum(weights) - sum(weights[:i])) for i in list(range(len(weights)))]
    idxList: list[int] = list(range(self._images.__len__()))
    samples: list[list[int]] = []
    for sampleProp in sampleProps:
        sampleSize = math.floor(sampleProp * len(idxList))
        sampleIdxList = rand.sample(idxList, k=sampleSize)
        idxList = list(set(idxList) - set(sampleIdxList))
        samples.append(sampleIdxList)
    assert sum([len(s) for s in samples]) == len(self._images)
    
    if showPbar:
        pbar = tqdm(total=sum([len(s) for s in samples]), unit="image(s)", leave=leavePbar)
        pbar.desc = "Splitting Datasets"

    datasets: list[DS] = []
    for sampleIdxList in samples:
        dataset = type(self)()
        dataset._info = self._info.copy()
        dataset._info.date_created = datetime.now()
        dataset._licenses = self._licenses.copy()
        dataset._categories = self._categories.copy()
        for idx in sampleIdxList:
            image = self._images[idx].copy()
            dataset._images.append(image)
            anns = self._annotations.search(lambda ann: ann.image_id == image.id)
            dataset._annotations._objects.extend(anns._objects.copy())
            if showPbar:
                pbar.update()
        dataset._images.sort(lambda img: img.id)
        dataset._annotations.sort(lambda ann: ann.id)
        datasets.append(dataset)
    if showPbar:
        pbar.close()
    return datasets