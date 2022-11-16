import numpy as np
import cv2
import numpy.typing as npt
from pyevu import BBox2D, Vector2
from .base import Base, BaseHandler
from .util import MaskUtil, CvUtil
from .polygon import Segmentation

class MaskSetting(Base):
    def __init__(
        self,
        track: bool=False,
        canOcclude: bool=True,
        canBeOccluded: bool=True,
        category: str="Mask", supercategory: str="Mask"
    ):
        self.track = track
        """Whether or not mask is added to handler."""

        self.canOcclude = canOcclude
        """Whether or not this mask can occlude existing masks in handler."""

        self.canBeOccluded = canBeOccluded
        """
        Whether or not this mask can be occluded by future masks.
        This is only relevant when isTracked=True.
        """

        self.category = category
        """COCO category used when converting to a coco dataset."""
        
        self.supercategory = supercategory
        """COCO supercategory used when converting to a coco dataset."""

    @property
    def skip(self) -> bool:
        return not self.track and not self.canOcclude

class Mask(Base): # TODO: Need an optional category label here for the sake of making datasets.
    def __init__(
        self, _mask: npt.NDArray[np.bool_]=None,
        setting: MaskSetting=MaskSetting()
    ):
        self._mask = _mask
        """_mask is set inside of drawing method."""

        self.setting = setting

    @property
    def track(self) -> bool:
        return self.setting.track

    @property
    def canOcclude(self) -> bool:
        return self.setting.canOcclude
    
    @property
    def canBeOccluded(self) -> bool:
        return self.setting.canBeOccluded
    
    @property
    def skip(self) -> bool:
        return self.setting.skip
    
    @property
    def bbox(self) -> BBox2D:
        if self._mask is None:
            return None
        y, x = np.where(self._mask)
        if len(x) == 0 or len(y) == 0:
            return None
        return BBox2D(Vector2(x.min().tolist(), y.min().tolist()), Vector2(x.max().tolist(), y.max().tolist()))

    @property
    def contours(self):
        contours, _   = cv2.findContours(self._mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    @property
    def segmentation(self) -> Segmentation:
        return Segmentation.from_contours(self.contours)

    def get_preview(self, showBBox: bool=False, showContours: bool=False, minNumPoints: int=None) -> npt.NDArray[np.uint8]:
        assert self._mask is not None, f"_mask isn't initialized yet"
        maskImg = np.zeros(tuple(list(self._mask.shape) + [3]))
        maskImg[self._mask] = (255, 255, 255)
        
        if showBBox:
            bbox = self.bbox
            if bbox is not None:
                maskImg = CvUtil.rectangle(
                    img=maskImg,
                    pt1=tuple(bbox.v0), pt2=tuple(bbox.v1),
                    color=(0, 0, 255),
                    thickness=2, lineType=cv2.LINE_AA
                )
        if showContours:
            if minNumPoints is None:
                contours = self.contours
                maskImg = cv2.drawContours(maskImg, contours, -1, (255, 0, 0), 2)
            else:
                seg = self.segmentation
                seg = seg.prune(lambda poly: len(poly) < minNumPoints)
                maskImg = cv2.drawContours(maskImg, seg.to_contours(), -1, (255, 0, 0), 2)
        return maskImg

    @property
    def preview(self) -> npt.NDArray[np.uint8]:
        return self.get_preview()

    def show_preview(self, showBBox: bool=False, showContours: bool=False, minNumPoints: int=None):
        maskImg = self.get_preview(showBBox=showBBox, showContours=showContours, minNumPoints=minNumPoints)

        cv2.imshow('mask preview', maskImg)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

class MaskHandler(BaseHandler[Mask]):
    def __init__(
        self, _objects: list[Mask]=None
    ):
        super().__init__(_objects)
    
    def process(self, mask: Mask):
        if mask.skip:
            return
        assert mask._mask is not None, "mask._mask hasn't been initialized yet."
        if len(self) > 0:
            assert self[0]._mask.shape == mask._mask.shape, f"New mask shape {mask._mask.shape} doesn't match exiting shape {self[0]._mask.shape}"
        if mask.canOcclude:
            for trackedMask in self:
                if trackedMask.canBeOccluded:
                    MaskUtil.occlude(mask._mask, trackedMask._mask)
        if mask.track:
            self.append(mask)

    @property
    def preview(self) -> npt.NDArray[np.uint8]:
        _mask: npt.NDArray[np.bool_] = None
        for mask in self:
            assert mask._mask is not None, f"_mask isn't initialized yet"
            if _mask is None:
                _mask = mask._mask.copy()
            else:
                _mask |= mask._mask
        maskImg = np.zeros(tuple(list(_mask.shape) + [3]))
        maskImg[_mask] = (255, 255, 255)
        return maskImg

    def show_preview(self):
        cv2.imshow('mask preview', self.preview)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
