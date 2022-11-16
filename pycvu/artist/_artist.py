from __future__ import annotations
from functools import partial
from typing import overload, Callable
import types
import cv2
import os
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from shutil import rmtree
import glob

from pycvu.base import BaseUtil, Base, Context, ContextVarRef
from ..color import Color
from ..mask import MaskSetting, MaskHandler
from ..util import CvUtil, Convert, \
    VectorVar, ImageVectorCallback, ColorVar, NoiseVar, \
    IntVar, FloatVar, StringVar, ImageVar, ImageInput, \
    LoadableImageMask, LoadableImageMaskHandler

__all__ = [
    "Artist"
]

from ._draw_process import DrawProcess, DrawProcessQueue, \
    DrawProcessGroup, DrawPreprocessQueue

from ..coco.object_detection import *
from datetime import datetime

class Artist(Base):
    context: Context = Context()

    color: ColorVar = Color(255, 255, 255)
    """Color used when drawing anything."""

    thickness: IntVar = 1
    """Line thickness used when drawing non-solid shapes."""

    lineType: int = cv2.LINE_AA
    """Line type used when drawing non-solid shapes."""

    interpolation: int = cv2.INTER_AREA
    """
    Interpolation used when resizing.
    This can be any cv2.INTER_* value.
    """

    fontFace: int = cv2.FONT_HERSHEY_COMPLEX
    """Font face usedd when drawing text."""
    
    fontScale: FloatVar = 1.0
    """Font scale used when drawing text."""

    maskSetting: MaskSetting = MaskSetting()
    """Use for controlling which masks should be tracked and/or contribute to occlusion."""

    from ._pil_artist import PilArtist as PIL

    def __init__(self, src: ImageInput | None=None):
        super().__init__()
        self._img: npt.NDArray[np.uint8] | None = None
        if src is not None:
            self._img = Convert.cast_image_input(src)
        self._drawQueue: DrawProcessQueue = DrawProcessQueue()

        self.pil = Artist.PIL(self)
        self.local = DrawPreprocessQueue()
        self.group = ProcessGrouper(self)
    
    @property
    def src(self) -> npt.NDArray[np.uint8] | None:
        return self._img
    
    @src.setter
    def src(self, value: ImageInput):
        if value is not None:
            self._img = Convert.cast_image_input(value)
        else:
            self._img = None

    @classmethod
    def get_meta(cls) -> dict:
        def _get_meta_dict(objCls, excludedKeys: list[str]=[]) -> dict:
            metaDict: dict = dict()
            for key, val in objCls.__dict__.items():
                if (
                    key.startswith('_') or key in excludedKeys
                    or type(val) in [
                        types.FunctionType,
                        classmethod,
                        staticmethod,
                        property
                    ]
                ):
                    continue
                if hasattr(val, 'to_dict'):
                    metaDict[key] = val.to_dict()
                else:
                    metaDict[key] = val
            return metaDict
        
        meta: dict = dict(
            cv=_get_meta_dict(cls, excludedKeys=['PIL']),
            pil=_get_meta_dict(cls.PIL)
        )
        return meta

    @classmethod
    def set_meta(cls, meta: dict):
        def _set_meta(metaObjCls, metaDict: dict):
            for key, val in metaDict.items():
                assert hasattr(metaObjCls, key)
                if type(val) is dict and '_typedict' in val:
                    objCls = BaseUtil.from_type_dict(val['_typedict'])
                    if hasattr(objCls, 'from_dict'):
                        obj = objCls.from_dict(val)
                    else:
                        obj = objCls(**{key0: val0 for key0, val0 in val.items() if key0 != '_typedict'})
                    setattr(metaObjCls, key, obj)
                else:
                    setattr(metaObjCls, key, val)
        
        assert type(meta) is dict, f"Expected meta of type dict. Encountered: {type(meta)}"
        assert 'cv' in meta
        assert 'pil' in meta

        _set_meta(cls, meta['cv'])
        _set_meta(cls.PIL, meta['pil'])

    def to_dict(self, saveImg: bool=True, saveMeta: bool=True) -> dict:
        itemDict: dict = dict()
        itemDict['_img'] = self._img.tolist() if saveImg and self._img is not None else None
        itemDict['_drawQueue'] = self._drawQueue.to_dict()
        itemDict['local'] = self.local.to_dict()
        meta = type(self).get_meta() if saveMeta else None
        itemDict['meta'] = meta
        return itemDict

    @classmethod
    def from_dict(cls, item_dict: dict, img: np.ndarray=None, loadMeta: bool=True) -> Artist:
        if img is None and item_dict['_img'] is None:
            pass
        elif img is None:
            img = np.array(item_dict['_img'], dtype=np.uint8)
        artist = Artist(img)
        if loadMeta and item_dict['meta'] is not None:
            cls.set_meta(item_dict['meta'])
        
        artist._drawQueue = DrawProcessQueue.from_dict(item_dict['_drawQueue'], context=cls.context)
        artist.local = DrawPreprocessQueue.from_dict(item_dict['local'])
        return artist

    def save(self, path: str, saveImg: bool=True, saveMeta: bool=True):
        super().save(path, saveImg=saveImg, saveMeta=saveMeta)

    @classmethod
    def load(cls, path: str, img: np.ndarray=None, loadMeta: bool=True) -> Artist:
        return super().load(path, img=img, loadMeta=loadMeta)

    def _add_process(self, dp: DrawProcess):
        if type(dp) is DrawProcess:
            self._drawQueue.append(dp)
        else:
            raise TypeError

    def circle(
        self, center: VectorVar | ImageVectorCallback,
        radius: IntVar, fill: bool=False,
        weight: float=1, repeat: int=1, prob: float=1.0
    ) -> Artist:
        """Draws a circle.

        Args:
            center (tuple[int, int] | Vector): Center of the circle.
            radius (int): Radius of the circle.
            fill (bool, optional): Whether or not to fill the circle in. Defaults to False.
        """
        p = partial(
            CvUtil.circle,
            center=center, radius=radius,
            color=Artist.color,
            thickness=Artist.thickness if not fill else -1,
            lineType=Artist.lineType
        )
        maskSetting = self.maskSetting.copy() if not self.maskSetting.skip else None
        self._add_process(DrawProcess(p, maskSetting, weight=weight, repeat=repeat, prob=prob))
        return self
    
    def ellipse(
        self, center: VectorVar | ImageVectorCallback,
        axis: VectorVar,
        angle: FloatVar=0,
        startAngle: FloatVar=0,
        endAngle: FloatVar=360,
        fill: bool=False,
        weight: float=1, repeat: int=1, prob: float=1.0
    ) -> Artist:
        """Draws an ellipse.

        Args:
            center (tuple[int, int] | Vector): Center of the ellipse.
            axis (tuple[int, int] | Vector): The a and b axes of the ellipse. Major and minor axis is not in any particular order.
            angle (float, optional): Angle by which the ellipse is rotated before drawing. Defaults to 0.
            startAngle (float, optional): The angle that you want to start drawing the ellipse at. Defaults to 0.
            endAngle (float, optional): The angle that you want to stop drawing the ellipse at. Defaults to 360.
            fill (bool, optional): Whether or not you want to fill in the ellipse when drawing. Defaults to False.
        """
        p = partial(
            CvUtil.ellipse,
            center=center, axis=axis,
            angle=angle, startAngle=startAngle, endAngle=endAngle,
            color=Artist.color,
            thickness=Artist.thickness if not fill else -1,
            lineType=Artist.lineType
        )
        maskSetting = self.maskSetting.copy() if not self.maskSetting.skip else None
        self._add_process(DrawProcess(p, maskSetting, weight=weight, repeat=repeat, prob=prob))

    def rectangle(
        self,
        pt1: VectorVar | ImageVectorCallback,
        pt2: VectorVar | ImageVectorCallback,
        fill: bool=False,
        rotation: FloatVar=0,
        weight: float=1,
        repeat: int=1, prob: float=1.0
    ) -> Artist:
        """Draws a rectangle.

        Args:
            pt1 (tuple[int, int] | Vector): First corner of rectangle.
            pt2 (tuple[int, int] | Vector): Second corner of rectangle.
            fill (bool, optional): Whether or not to fill in the rectangle when drawing. Defaults to False.
        """
        p = partial(
            CvUtil.rectangle,
            pt1=pt1, pt2=pt2,
            color=Artist.color,
            thickness=Artist.thickness if not fill else -1,
            lineType=Artist.lineType,
            rotation=rotation
        )
        maskSetting = self.maskSetting.copy() if not self.maskSetting.skip else None
        self._add_process(DrawProcess(p, maskSetting, weight=weight, repeat=repeat, prob=prob))
        return self
    
    def line(
        self,
        pt1: VectorVar | ImageVectorCallback,
        pt2: VectorVar | ImageVectorCallback,
        weight: float=1, repeat: int=1, prob: float=1.0
    ) -> Artist:
        """Draws a line.

        Args:
            pt1 (tuple[int, int] | Vector): Where to start drawing the line.
            pt2 (tuple[int, int] | Vector): Where to stop drawing the line.
        """
        p = partial(
            CvUtil.line,
            pt1=pt1, pt2=pt2,
            color=Artist.color,
            thickness=Artist.thickness,
            lineType=Artist.lineType
        )
        maskSetting = self.maskSetting.copy() if not self.maskSetting.skip else None
        self._add_process(DrawProcess(p, maskSetting, weight=weight, repeat=repeat, prob=prob))
        return self
    
    def affine_rotate(
        self, angle: FloatVar,
        degrees: bool=True,
        scale: FloatVar=1,
        adjustBorder: bool=False,
        center: VectorVar=None,
        weight: float=1, prob: float=1.0
    ) -> Artist:
        """This rotates the image about the axis coming out of the screen.

        Args:
            angle (float): angle of rotation
            degrees (bool, optional): Whether or not the angle is given in degrees. Defaults to True.
            scale (float, optional):
                Scaling factor to apply to the image after being rotated.
                Without rescaling the image projection, parts of the original image may be cut off.
                Alternatively, you can use adjustBorder=True to automatically adjust the scale to fit the window perfectly.
                Defaults to 1.
            adjustBorder (bool, optional):
                If set to True, the image projection will automatically be rescaled such
                that the image fits perfectly in the window.
                Defaults to False.
            center (tuple[int, int] | Vector, optional): The center of rotation. Defaults to center of the image.
        """
        p = partial(
            CvUtil.affine_rotate,
            angle=angle, degrees=degrees,
            scale=scale, interpolation=Artist.interpolation,
            adjustBorder=adjustBorder, center=center,
            borderColor=Artist.color
        )
        self._add_process(DrawProcess(p, weight=weight, prob=prob))
        return self

    def text(
        self, text: StringVar,
        org: VectorVar,
        bottomLeftOrigin: bool=False,
        rotation: FloatVar=0,
        weight: float=1, repeat: int=1, prob: float=1.0
    ) -> Artist:
        """Draws text on the image.

        Args:
            text (str): The text string that you would like to draw.
            org (tuple[int, int] | Vector): Where the text should be drawn.
            bottomLeftOrigin (bool, optional):
                In OpenCV coordinates, (0,0) is assumed to be at the upper-left
                corner of the image. Setting bottomLeftOrigin=True changes the
                assumption to be the bottom-left corner of the image.
                Defaults to False.
        """
        p = partial(
            CvUtil.text,
            text=text, org=org,
            fontFace=Artist.fontFace, fontScale=Artist.fontScale,
            color=Artist.color, thickness=Artist.thickness,
            lineType=Artist.lineType,
            bottomLeftOrigin=bottomLeftOrigin,
            rotation=rotation
        )
        maskSetting = self.maskSetting.copy() if not self.maskSetting.skip else None
        self._add_process(DrawProcess(p, maskSetting, weight=weight, repeat=repeat, prob=prob))

    def overlay_image(
        self, foreground: ImageVar,
        position: VectorVar | ImageVectorCallback,
        rotation: FloatVar=0, scale: FloatVar=1,
        noise: NoiseVar=None,
        weight: float=1, repeat: int=1, prob: float=1.0
    ) -> Artist:
        p = partial(
            CvUtil.overlay_image,
            foreground=foreground,
            position=position,
            rotation=rotation,
            scale=scale,
            noise=noise
        )
        maskCompatibleTypes = [LoadableImageMask, LoadableImageMaskHandler]
        if (
            not self.maskSetting.skip
            and (
                type(foreground) in maskCompatibleTypes
                or (
                    type(foreground) is ContextVarRef
                    and type(foreground.value) in maskCompatibleTypes
                )
            )
        ):
            maskSetting = self.maskSetting.copy()
        else:
            maskSetting = None
        self._add_process(DrawProcess(p, maskSetting, weight=weight, repeat=repeat, prob=prob))

    @overload
    def resize(self, dsize: VectorVar, weight: float=1, prob: float=1.0) -> Artist:
        """Resizes the working image to the given target size.

        Args:
            dsize (tuple[int, int] | Vector): target size
        """
        ...

    @overload
    def resize(self, fx: FloatVar=None, fy: FloatVar=None, weight: float=1, prob: float=1.0) -> Artist:
        """Resizes the working image according to the given scaling factors.
        Scaling factors are relative to the current working image size.

        Args:
            fx (float, optional): x resize scaling factor. Defaults to None.
            fy (float, optional): y resize scaling factor. Defaults to None.
        """
        ...

    def resize(
        self, dsize: VectorVar=None,
        fx: FloatVar=None, fy: FloatVar=None,
        weight: float=1, prob: float=1.0
    ) -> Artist:
        p = partial(
            CvUtil.resize,
            dsize=dsize,
            fx=fx, fy=fy,
            interpolation=Artist.interpolation
        )
        self._add_process(DrawProcess(p, weight=weight, prob=prob))

    def draw(self) -> np.ndarray:
        """Perform all of the actions in the drawing queue on the source image and return the result.
        """
        assert self._img is not None, f"Need to specify src before drawing."
        result = self._img.copy()
        result = self._drawQueue.draw(
            img=result, preprocess=self.local.preprocess
        )
        return result
    
    def draw_and_get_masks(self) -> tuple[np.ndarray, MaskHandler]:
        """Perform all of the actions in the drawing queue on the source image and return the result.
        """
        assert self._img is not None, f"Need to specify src before drawing."
        result = self._img.copy()
        result, maskHandler = self._drawQueue.draw_and_get_masks(
            img=result, preprocess=self.local.preprocess
        )
        return result, maskHandler

    def draw_and_save(self, path: str):
        """Perform all of the actions in the drawing queue on the source image and save the result to a file.
        """
        cv2.imwrite(path, self.draw())

    def _generate_dataset(
        self, frames: int, dumpDir: str="artist_dataset_dump",
        showPbar: bool=True
    ):
        if os.path.isdir(dumpDir):
            rmtree(dumpDir)
        os.makedirs(dumpDir, exist_ok=True)
    
        dataset = Dataset()
        dataset.info.description = 'Generated with pycvu Artist.'
        dataset.info.date_created = datetime.now()
        dataset.info.year = datetime.now().year
        dataset.licenses.append(
            License(id=0, name='No License', url='N/A')
        )
        pbar = tqdm(total=frames, leave=True) if showPbar else None
        if pbar is not None:
            pbar.set_description("Generating Dataset")
        for i in range(frames):
            result, maskHandler = self.draw_and_get_masks()
            imgPath = f"{dumpDir}/frame{i}.png"
            cv2.imwrite(imgPath, result)
            image = Image(
                id=len(dataset.images),
                width=result.shape[1], height=result.shape[0],
                file_name=imgPath, date_captured=datetime.now()
            )
            dataset.images.append(image)

            for j, mask in enumerate(maskHandler):
                if mask._mask.sum() == 0:
                    continue

                category = dataset.categories.get(
                    lambda c: c.name == mask.setting.category
                    and c.supercategory == mask.setting.supercategory
                )
                if category is None:
                    category = Category(
                        id=len(dataset.categories),
                        name=mask.setting.category,
                        supercategory=mask.setting.supercategory
                    )
                    dataset.categories.append(category)
                
                bbox = mask.bbox
                seg = mask.segmentation
                seg = seg.prune(lambda poly: len(poly) < 3)
                ann = Annotation(
                    id=len(dataset.annotations),
                    image_id=image.id, category_id=category.id,
                    segmentation=seg.coco,
                    area=bbox.area,
                    bbox=[
                        bbox.v0.x, bbox.v0.y,
                        bbox.xInterval.length, bbox.yInterval.length
                    ],
                    iscrowd=0
                )
                dataset.annotations.append(ann)
            if pbar is not None:
                pbar.update()
        dataset.save(f"{dumpDir}/dataset.json")

    def generate_dataset(
        self, frames: int, dumpDir: str="artist_dataset_dump",
        showPbar: bool=True, repeat: int=1
    ):
        assert len(self._drawQueue) > 0, f"Nothing has been queued for drawing yet."
        if repeat > 1:
            if not os.path.isdir(dumpDir):
                os.makedirs(dumpDir)
                currentIter = 0
            else:
                datasetDirPaths = [
                    path
                    for path in glob.glob(f"{dumpDir}/dataset*")
                    if os.path.isdir(path)
                ]
                iterNums = [int(os.path.basename(path).replace('dataset', '')) for path in datasetDirPaths]
                lastIter = max(iterNums)
                lastDatasetDir = datasetDirPaths[iterNums.index(lastIter)]
                if os.path.isfile(f"{lastDatasetDir}/dataset.json"):
                    rmtree(lastDatasetDir)
                    currentIter = lastIter
                else:
                    currentIter = lastIter + 1
            for k in range(currentIter, repeat):
                kStr = str(k)
                while len(kStr) < 3:
                    kStr = f"0{kStr}"
                self._generate_dataset(
                    frames=frames, dumpDir=f"{dumpDir}/dataset{kStr}",
                    showPbar=showPbar
                )
        else:
            self._generate_dataset(
                frames=frames, dumpDir=dumpDir,
                showPbar=showPbar
            )

    from ._debug import debug, debug_loop, group_debug

class ProcessGrouper:
    """
    Groups drawing methods together.
    This is mainly useful for shuffling the order of drawing processes.
    Meta settings are reverted after exiting, so it is also useful for creating
    a local meta scope.
    """
    def __init__(self, rootArtist: Artist):
        self._rootArtist = rootArtist
        self._groupArtists: list[Artist] = []
        self._metaStates: list[dict] = []

        self._pendingWeight: float = 1.0
        self._pendingRepeat: IntVar = 1
        self._pendingProb: float = 1.0
        self._pendingPreprocOps: list[Callable[[DrawPreprocessQueue],]] = []
        self._preprocs: list[DrawPreprocessQueue] = []

    def __call__(self, weight: float=1.0, repeat: IntVar=1, prob: float=1.0) -> ProcessGrouper:
        self._pendingWeight = weight
        self._pendingRepeat = repeat
        self._pendingProb = prob
        return self

    def shuffle(self) -> ProcessGrouper:
        def _shuffle(queue: DrawPreprocessQueue):
            queue.shuffle()
        self._pendingPreprocOps.append(_shuffle)
        return self
    
    def one_of(self) -> ProcessGrouper:
        def _one_of(queue: DrawPreprocessQueue):
            queue.one_of()
        self._pendingPreprocOps.append(_one_of)
        return self
    
    def some_of(self, n: IntVar) -> ProcessGrouper:
        def _some_of(queue: DrawPreprocessQueue):
            queue.some_of(n=n)
        self._pendingPreprocOps.append(_some_of)
        return self

    def __enter__(self) -> Artist:
        # print(f"{type(self).__name__} __enter__")
        groupArtist = Artist()
        for preproc_op in self._pendingPreprocOps:
            preproc_op(groupArtist.local)
        self._pendingPreprocOps.clear()

        self._groupArtists.append(groupArtist)
        self._metaStates.append(self._rootArtist.get_meta())
        self._preprocs.append(DrawPreprocessQueue())
        return groupArtist
    
    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        # print(f"{type(self).__name__} __exit__, {exc_type=}, {exc_val=}, {exc_tb=}")
        groupArtist = self._groupArtists.pop()
        meta = self._metaStates.pop()
        preproc = self._preprocs.pop()
        processes = groupArtist._drawQueue._objects.copy()
        preproc._opQueue.extend(groupArtist.local._opQueue.copy())

        del groupArtist
        if len(processes) > 0:
            group = DrawProcessGroup(processes, _preproc=preproc)
            self._rootArtist._add_process(
                DrawProcess(
                    group,
                    weight=self._pendingWeight,
                    repeat=self._pendingRepeat,
                    prob=self._pendingProb
                )
            )
        self._rootArtist.set_meta(meta)
        self._pendingWeight = 1.0
        self._pendingRepeat = 1
        self._pendingProb = 1.0
        self._pendingPreprocOps.clear()
