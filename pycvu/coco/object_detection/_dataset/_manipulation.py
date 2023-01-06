from __future__ import annotations
from typing import Callable
from datetime import datetime
import copy
from tqdm import tqdm
from ..._format import CocoBase, CocoBaseHandler, Image, License
from .._structs import Annotation, Category

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Dataset

@classmethod
def combine(cls: type[Dataset], sources: list[Dataset | str], showPbar: bool=False) -> Dataset:
    combined = cls()
    
    # Process Info
    combined.info.description = "Combined dataset using pycvu."
    combined.info.date_created = datetime.now()
    combined.info.year = datetime.now().year

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
        
        innerPbar = tqdm(total=len(dataset.images), leave=False) if showPbar else None
        dataset.images.sort(lambda image: image.id)
        for image in dataset.images:
            # Process License
            license = dataset.licenses.get(lambda lic: lic.id == image.license)
            assert license is not None
            lic = combined.licenses.get(
                lambda lic: all([
                    getattr(lic, key) == getattr(license, key)
                    for key in license.__dict__
                    if key != 'id'
                ])
            )
            if lic is None:
                # New license.
                lic = license.copy()
                lic.id = len(combined.licenses)
                combined.licenses.append(lic)

            # Process Image
            img = image.copy()
            img.id = len(combined.images)
            img.license = lic.id
            combined.images.append(img)

            for annotation in dataset.annotations.search(lambda ann: ann.image_id == image.id):
                # Process Category
                category = dataset.categories.get(lambda cat: cat.id == annotation.category_id)
                assert category is not None
                cat = combined.categories.get(
                    lambda cat: all([
                        getattr(cat, key) == getattr(category, key)
                        for key in category.__dict__
                        if key != 'id'
                    ])
                )
                if cat is None:
                    # New category.
                    cat = category.copy()
                    cat.id = len(combined.categories)
                    combined.categories.append(cat)

                # Process Annotation
                ann = annotation.copy()
                ann.id = len(combined.annotations)
                ann.image_id = img.id
                ann.category_id = cat.id
                combined.annotations.append(ann)
            if showPbar:
                innerPbar.update()
        if showPbar:
            innerPbar.close()
            outerPbar.update()
    if showPbar:
        outerPbar.close()
    return combined

def filter(
    self: Dataset,
    catFilter: Callable[[Category], bool]=None,
    annFilter: Callable[[Annotation], bool]=None,
    imgFilter: Callable[[Image], bool]=None,
    licFilter: Callable[[License], bool]=None,
    cleanupCat: bool=True, cleanupAnn: bool=True, cleanupImg: bool=True, cleanupLic: bool=True,
    reindex: bool=False,
    showPbar: bool=False, leavePbar: bool=False,
    applyToSelf: bool=False
) -> Dataset:
    # TODO: docstring
    # Note: Each filter affects all handlers either directly or through broken references.
    # Example: catFilter affects categories directly, and affects annotations, images, and
    #          licenses through broken references.
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
        (result.categories, catFilter),
        (result.annotations, annFilter),
        (result.images, imgFilter),
        (result.licenses, licFilter)
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
            pbar = tqdm(total=len(result.annotations), leave=leavePbar)
            pbar.set_description("Cleanup Annotations w/o Categories")
        for idx in list(range(len(result.annotations)))[::-1]:
            ann = result.annotations[idx]
            if result.categories.get(id=ann.category_id) is None:
                del result.annotations[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if licFilter is not None and cleanupImg:
        # Licenses affect Images
        if showPbar:
            pbar = tqdm(total=len(result.images), leave=leavePbar)
            pbar.set_description("Cleanup Images w/o Licenses")
        for idx in list(range(len(result.images)))[::-1]:
            img = result.images[idx]
            if result.licenses.get(id=img.license) is None:
                del result.images[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if annFilter is not None or catFilter is not None and cleanupImg:
        # Annotations affect Images -> Images Done
        if showPbar:
            pbar = tqdm(total=len(result.images), leave=leavePbar)
            pbar.set_description("Cleanup Images w/o Annotations")
        for idx in list(range(len(result.images)))[::-1]:
            img = result.images[idx]
            if result.annotations.get(image_id=img.id) is None:
                del result.images[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if imgFilter is not None or annFilter is not None or catFilter is not None and cleanupLic:
        # Images affect Licenses -> Licenses Done
        if showPbar:
            pbar = tqdm(total=len(result.licenses), leave=leavePbar)
            pbar.set_description("Cleanup Licenses w/o Images")
        for idx in list(range(len(result.licenses)))[::-1]:
            lic = result.licenses[idx]
            if result.images.get(license=lic.id) is None:
                del result.licenses[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if imgFilter is not None or licFilter is not None and cleanupAnn:
        # Images affect Annotations -> Annotations Done
        if showPbar:
            pbar = tqdm(total=len(result.annotations), leave=leavePbar)
            pbar.set_description("Cleanup Annotations w/o Images")
        for idx in list(range(len(result.annotations)))[::-1]:
            ann = result.annotations[idx]
            if result.images.get(id=ann.image_id) is None:
                del result.annotations[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
    if annFilter is not None or imgFilter is not None or licFilter is not None and cleanupCat:
        # Annotations affect Categories -> Categories Done
        if showPbar:
            pbar = tqdm(total=len(result.categories), leave=leavePbar)
            pbar.set_description("Cleanup Categories w/o Annotations")
        for idx in list(range(len(result.categories)))[::-1]:
            cat = result.categories[idx]
            if result.annotations.get(category_id=cat.id) is None:
                del result.categories[idx]
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()

    if reindex:
        result.reindex(applyToSelf=True, showPbar=showPbar, leavePbar=leavePbar)

    return result

def reindex(
    self: Dataset, applyToSelf: bool=True,
    showPbar: bool=False, leavePbar: bool=False
) -> Dataset:
    if not applyToSelf:
        result = copy.deepcopy(self)
    else:
        result = self
    
    def updateLicenseIdInImages(oldId: int, newId: int):
        for img in result.images.search(lambda img: img.license == oldId):
            img.license = newId

    def updateImgIdInAnnotations(oldId: int, newId: int):
        for ann in result.annotations.search(lambda ann: ann.image_id == oldId):
            ann.image_id = newId
    
    def updateCatIdInAnnotations(oldId: int, newId: int):
        for ann in result.annotations.search(lambda ann: ann.category_id == oldId):
            ann.category_id = newId

    for handler, idUpdateCallback in [
        (result.licenses, updateLicenseIdInImages),
        (result.images, updateImgIdInAnnotations),
        (result.categories, updateCatIdInAnnotations),
        (result.annotations, None)
    ]:
        handler: CocoBaseHandler = handler
        handler.reindex(
            showPbar=showPbar, leavePbar=leavePbar,
            applyToSelf=True,
            idUpdateCallback=idUpdateCallback
        )
    
    return result
