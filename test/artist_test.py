from __future__ import annotations
import os
import numpy as np
from pycvu import Artist, Interval, Color, HSV, Vector
from pycvu.util import LoadableImageMaskHandler, \
    CvUtil, PilUtil, DebugTimer
from pycvu.text_generator import TextGenerator, TextSampler, \
    CharacterSets, StringSets
from pycvu.base import Base, ContextVarRef

class LabelManager(Base):
    def __init__(
        self,
        categoryLabelPathMap: dict[str, str]=None,
        categoryImgHandlerMap: dict[str, LoadableImageMaskHandler]=None
    ):
        if categoryLabelPathMap is None:
            self.categoryLabelPathMap: dict[str, str] = {
                'label0': 'symbol/*_label0_*.png',
                'label1': 'symbol/*_label1_*.png',
                'label3': 'symbol/*_label3.png',
                'label4': 'symbol/*_label4.png',
                'label5': 'symbol/*_label5.png',
                'label6': 'symbol/*_label6.png',
                'label7': 'symbol/*_label7.png',
                'label14': 'symbol/*_label14_*.png',
                'bg': 'symbol/bg_*.png'
            }
        else:
            self.categoryLabelPathMap = categoryLabelPathMap
        
        if categoryImgHandlerMap is None:
            maskThresh = Interval[HSV](HSV(0,0,0), HSV(359.9, 1, 220/255))
            self.categoryImgHandlerMap: dict[str, LoadableImageMaskHandler] = {
                category: LoadableImageMaskHandler.from_wildcard(wildcardPath, maskThresh)
                for category, wildcardPath in self.categoryLabelPathMap.items()
            }
            for category in self.categoryImgHandlerMap:
                self.categoryImgHandlerMap[category].load_data()
        else:
            self.categoryImgHandlerMap = categoryImgHandlerMap

    def to_dict(self) -> dict:
        return dict(
            categoryLabelPathMap=self.categoryLabelPathMap,
            categoryImgHandlerMap={
                category: imgHandler.to_dict()
                for category, imgHandler in self.categoryImgHandlerMap.items()
            }
        )

    @classmethod
    def from_dict(cls, item_dict: dict) -> LabelManager:
        return LabelManager(
            categoryLabelPathMap=item_dict['categoryLabelPathMap'],
            categoryImgHandlerMap={
                key: LoadableImageMaskHandler.from_dict(val)
                for key, val in item_dict['categoryImgHandlerMap'].items()
            }
        )
    
    def register_images(self) -> dict[str, ContextVarRef]:
        return {key: Artist.context.register_variable(val) for key, val in self.categoryImgHandlerMap.items()}

with DebugTimer.probe("Label Manager Init"):
    labelManagerPath = 'labelManager.json'
    if not os.path.isfile(labelManagerPath):
        labelManager = LabelManager()
        labelManager.save(labelManagerPath)
    else:
        labelManager = LabelManager.load(labelManagerPath)
    registeredImages: dict[str, ContextVarRef] = labelManager.register_images()

with DebugTimer.probe("Artist Init"):
    artist = Artist(np.ones((800, 800, 3), dtype=np.uint8) * 255)

    # Proc 0
    Artist.fontScale = Interval[float](0.4, 0.5)
    Artist.thickness = Interval[int](1, 2)
    # Artist.color = Interval[Color](Color(0, 0, 0), Color(50, 50, 50)) # hsv would be better
    # Kume was setting all color channels to the same value. Doing this with hsv should have the same effect.
    Artist.color = Interval[HSV](HSV(0,0,0), HSV(0,0,50/255))
    artist.text(
        text=TextGenerator(
            characters=CharacterSets.alpha,
            textLength=1, # n_connect
            allowRepetition=True
        ),
        org=CvUtil.Callback.get_position_interval,
        repeat=50 # iters
    )

    # Proc 1
    Artist.PIL.fontSize = Interval[int](10, 15)
    artist.pil.text(
        text=TextGenerator(
            characters=CharacterSets.kana,
            textLength=1, # n_connect
            allowRepetition=True
        ),
        position=PilUtil.Callback.get_position_interval,
        rotation=Interval[float](-20, 20),
        repeat=50 # iters
    )

    # Proc 2
    artist.pil.text(
        text=TextGenerator(
            characters=CharacterSets.kanji,
            textLength=1, # n_connect
            allowRepetition=True
        ),
        position=PilUtil.Callback.get_position_interval,
        rotation=Interval[float](-20, 20),
        repeat=50 # iters
    )

    # Proc 2
    artist.pil.text(
        text=TextGenerator(
            characters=CharacterSets.kanji,
            textLength=Interval[int](2, 10), # Kume had this fixed at 10
            allowRepetition=True
        ),
        position=PilUtil.Callback.get_position_interval,
        rotation=Interval[float](-5, 5),
        repeat=10 # iters
    )

    Artist.maskSetting.track = True
    Artist.maskSetting.supercategory = "signature"
    Artist.maskSetting.category = "name"

    # Proc 3
    artist.pil.text(
        text=TextSampler(StringSets.namae),
        position=PilUtil.Callback.get_position_interval,
        direction='rtl',
        rotation=Interval[float](-20, 20),
        repeat=50 # iters
    )

    # Proc 4
    artist.pil.text(
        text=TextSampler(StringSets.namae),
        position=PilUtil.Callback.get_position_interval,
        direction='ttb',
        rotation=Interval[float](-20, 20),
        repeat=20 # iters
    )

    # Proc 5
    Artist.maskSetting.category = "hanko"
    Artist.PIL.fontSize = Interval[int](15, 30)
    Artist.PIL.hankoIsVertical = True
    Artist.PIL.hankoMarginRatio = Interval[float](0.0, 0.2)
    Artist.PIL.hankoOutlineWidthRatio = Interval[float](0.05, 0.3)
    artist.pil.hanko(
        text=TextSampler(StringSets.namae),
        position=PilUtil.Callback.get_position_interval,
        rotation=Interval[float](-20, 20),
        repeat=20 # iters
    )

    Artist.maskSetting.track = False

    # Proc 6 & 7
    artist.line(
        pt1=CvUtil.Callback.get_position_interval,
        pt2=CvUtil.Callback.get_position_interval,
        repeat=1+3
    )

    # Proc 8
    artist.circle(
        center=CvUtil.Callback.get_position_interval,
        radius=Interval[int](10, 50),
        repeat=30
    )

    # Proc 9

    artist.ellipse(
        center=CvUtil.Callback.get_position_interval,
        axis=Interval[Vector[int]](Vector[int](100, 100), Vector[int](200, 200)), # Kume called this 'scale'. Incoherent naming.
        angle=Interval[float](-20, 20), # rotation
        repeat=0 # Removed from config??
    )

    # Proc 10
    Artist.thickness = Interval[int](1, 3)
    artist.rectangle(
        pt1=CvUtil.Callback.get_position_interval,
        pt2=CvUtil.Callback.get_position_interval,
        rotation=Interval[float](-10, 10),
        repeat=10
    )

    Artist.maskSetting.track = True
    Artist.maskSetting.supercategory = "waku"
    Artist.maskSetting.category = "waku"
    # Proc 11
    artist.overlay_image(
        # foreground=labelImgHandler[0],
        foreground=registeredImages['label0'],
        position=CvUtil.Callback.get_position_interval,
        rotation=Interval[float](-5, 5),
        scale=Interval[float](0.9, 1.1),
        noise=Interval[int](-10, 20),
        repeat=1
    )

    # Proc 12
    artist.overlay_image(
        # foreground=labelImgHandler[1],
        foreground=registeredImages['label1'],
        position=CvUtil.Callback.get_position_interval,
        rotation=Interval[float](-5, 5),
        scale=Interval[float](0.9, 1.1),
        noise=Interval[int](-10, 20),
        repeat=1
    )

    # Proc 12
    label03to07 = LoadableImageMaskHandler()
    for label in ['label3', 'label4', 'label5', 'label6', 'label7']:
        label03to07 += labelManager.categoryImgHandlerMap[label]
    label03to07ref = Artist.context.register_variable(label03to07)
    artist.overlay_image(
        # foreground=labelImgHandler[3:7],
        foreground=label03to07ref,
        position=CvUtil.Callback.get_position_interval,
        rotation=Interval[float](-5, 5),
        scale=Interval[float](0.9, 1.1),
        noise=Interval[int](-10, 20),
        repeat=1
    )

    # Proc 13
    artist.overlay_image(
        # foreground=labelImgHandler[10],
        foreground=registeredImages['label14'],
        position=CvUtil.Callback.get_position_interval,
        rotation=Interval[float](-5, 5),
        scale=Interval[float](0.9, 1.1),
        noise=Interval[int](-20, 0),
        repeat=2
    )

    # Proc 14
    artist.overlay_image(
        # foreground=labelImgHandler[10],
        foreground=registeredImages['label14'],
        position=CvUtil.Callback.get_position_interval,
        rotation=Interval[float](-5, 5),
        scale=Interval[float](0.9, 1.1),
        noise=Interval[int](30, 50),
        repeat=1
    )

    Artist.maskSetting.track = False

    # Proc 15
    # type: "dest"
    # What is this?
    # It looks like it's the exact same as 'label', except that the coco annotations aren't calculated. Again, bad naming conventions in Kume's code.
    artist.overlay_image(
        # foreground=bgImgHandlerRef,
        foreground=registeredImages['bg'],
        position=CvUtil.Callback.get_position_interval,
        rotation=Interval[float](-20, 20),
        scale=Interval[float](0.8, 1.2),
        noise=Interval[int](-50, 100),
        repeat=2
    )

    numProc = len(artist._drawQueue)
    artist.local.some_of(Interval[int](min(5, numProc), min(10, numProc)))
    artist.local.shuffle()

    artist.save('artistTestSave.json', saveImg=False, saveMeta=True)

from shutil import rmtree
import cv2
from pycvu.coco.object_detection import *
from datetime import datetime

previewDump = 'artistTestDump'
if os.path.isdir(previewDump):
    rmtree(previewDump)
os.makedirs(previewDump, exist_ok=True)

info = Info(description='debug test dataset', date_created=datetime.now(), year=datetime.now().year)
license = License(id=0, name="No License", url='N/A')
licenses = Licenses([license])
images = Images()
annotations = Annotations()
categories = Categories()

batch_size: int = 10
with DebugTimer.probe(f"Dump Data Batch of {batch_size}"):
    for i in range(batch_size):
        with DebugTimer.probe(f"({i+1}/{batch_size}) Draw & Get Masks"):
            result, maskHandler = artist.draw_and_get_masks()
            # result = artist.draw()
        imgPath = f"{previewDump}/result{i}.png"
        cv2.imwrite(imgPath, result)
        cv2.imwrite(f"{previewDump}/maskPreview{i}.png", maskHandler.preview)
        image = Image(
            id=len(images), width=result.shape[1], height=result.shape[0],
            file_name=imgPath, date_captured=datetime.now()
        )
        images.append(image)

        for j, mask in enumerate(maskHandler):
            if mask._mask.sum() == 0:
                continue
            
            category = categories.get(
                lambda c: c.name == mask.setting.category
                and c.supercategory == mask.setting.supercategory
            )
            if category is None:
                category = Category(
                    id=len(categories),
                    name=mask.setting.category,
                    supercategory=mask.setting.supercategory
                )
                categories.append(category)

            minNumPoints = 3
            bbox = mask.bbox
            seg = mask.segmentation
            seg = seg.prune(lambda poly: len(poly) < minNumPoints)
            ann = Annotation(
                id=len(annotations),
                image_id=image.id, category_id=category.id,
                segmentation=seg.coco,
                area=bbox.area,
                bbox=[bbox.v0.x, bbox.v0.y, bbox.xInterval.length, bbox.yInterval.length],
                iscrowd=0
            )
            annotations.append(ann)
            maskImg = mask.get_preview(showBBox=True, showContours=True, minNumPoints=minNumPoints)
            numStr = str(j)
            while len(numStr) < 2:
                numStr = f"0{numStr}"
            maskPath = f"{previewDump}/mask{i}-{numStr}.png"
            cv2.imwrite(maskPath, maskImg)

dataset = Dataset(info=info, images=images, licenses=licenses, annotations=annotations, categories=categories)
dataset.save(f"{previewDump}/dataset.json")

# cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('result', 500, 500)
# cv2.imshow('result', result)
# cv2.waitKey()
# cv2.destroyAllWindows()
