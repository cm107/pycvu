from __future__ import annotations
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from pycvu.coco.object_detection import *
from pycvu.base import BaseHandler
import cv2
import numpy as np
import os
import glob
from shutil import rmtree
from tqdm import tqdm
from pyevu import Vector2, BBox2D
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from enum import Enum
from datetime import datetime
from ._config import EvalConfig

class CacheMode(Enum):
    READONLY = 0
    NEW = 1
    UPDATE = 2

class ConfusionType(Enum):
    TP_RESULT = 0
    FP_RESULT = 1
    TP_ANN = 2
    FN_ANN = 2

class EvalSession:
    def __init__(self, evalConfig: EvalConfig, cacheRoot: str='evalCache'):
        self.evalConfig = evalConfig

        os.makedirs(cacheRoot, exist_ok=True)
        cacheDirs = [path for path in glob.glob(f"{cacheRoot}/*") if os.path.isdir(path) and os.path.isfile(f"{path}/config.json")]
        matchedCacheDir: str = None
        for cacheDir in cacheDirs:
            existingEvalConfig = EvalConfig.load(f"{cacheDir}/config.json")
            if existingEvalConfig == self.evalConfig:
                matchedCacheDir = cacheDir
                break
        
        if matchedCacheDir is not None:
            self.cacheDir = matchedCacheDir
        else:
            self.cacheDir = f"{cacheRoot}/{int(datetime.now().timestamp() * 10**6)}"
            os.mkdir(self.cacheDir)
            self.evalConfig.save(f"{self.cacheDir}/config.json")
        print(f"Cache directory: {self.cacheDir}")

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(evalConfig.modelZooConfig))
        self.cfg.MODEL.WEIGHTS = evalConfig.weights
        self.cfg.INPUT.MIN_SIZE_TEST = evalConfig.minSizeTest
        self.cfg.INPUT.MAX_SIZE_TEST = evalConfig.maxSizeTest
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(evalConfig.names)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = evalConfig.scoreThreshTest

        self.cfg.MODEL.MASK_ON     = False
        self.cfg.MODEL.KEYPOINT_ON = False
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 0

        self.predictor = DefaultPredictor(self.cfg)
        
        self.gt: Dataset = None
        self.dt: Results = None
        
    def load_gt(self, cacheMode: CacheMode=CacheMode.NEW):
        cachePath = f'{self.cacheDir}/gt.json'
        if cacheMode in [CacheMode.NEW, CacheMode.UPDATE]:
            self.gt = Dataset.from_labelme(self.evalConfig.gtLabelmeDir, allowNoAnn=True)
            self.gt.filter(
                catFilter=lambda cat: cat.name == self.evalConfig.targetName,
                reindex=True, applyToSelf=True, cleanupImg=False
            )
            self.gt.save(cachePath)
        elif cacheMode == CacheMode.READONLY:
            self.gt = Dataset.load(cachePath)
        else:
            raise ValueError

    def calc_dt(self, cacheMode: CacheMode=CacheMode.UPDATE):
        if self.gt is None:
            self.load_gt()
        cachePath = f'{self.cacheDir}/dt.json'
        if cacheMode == CacheMode.READONLY:
            self.dt = Results.load(cachePath)
            if not os.path.isfile(f"{self.cacheDir}/cleanDt.json"):
                cleanDt = self.dt.search(lambda r: r.bbox is not None and r.score is not None)
                cleanDt.save(f"{self.cacheDir}/cleanDt.json")
        elif cacheMode in [CacheMode.NEW, CacheMode.UPDATE]:
            if cacheMode == CacheMode.NEW:
                self.dt = Results()
                images = self.gt.images
            elif cacheMode == CacheMode.UPDATE:
                if os.path.isfile(cachePath):
                    self.dt = Results.load(cachePath)
                else:
                    self.dt = Results()
                # Loop through new images only.
                images = self.gt.images.search(lambda image: self.dt.get(image_id=image.id) is None)
            else:
                raise ValueError
            
            for image in tqdm(images, desc="Evaluating GT"):
                img = cv2.imread(image.file_name)
                h, w = img.shape[:2]
                img[:int(h*0.75), :int(w*0.75), :] = (255, 255, 255)
                outputs = self.predictor(img)
                instances = outputs["instances"].to("cpu")

                noRelevantDet: bool = True
                for box, score, catId in zip(
                    instances._fields['pred_boxes'].tensor.tolist(),
                    instances._fields['scores'].tolist(),
                    instances._fields['pred_classes'].tolist()
                ):
                    box = [int(val) for val in box]
                    bbox = BBox2D(Vector2(*box[:2]), Vector2(*box[2:]))
                    result = BBoxResult(
                        image_id=image.id, category_id=catId,
                        bbox=[bbox.v0.x, bbox.v0.y, bbox.xInterval.length, bbox.yInterval.length],
                        score=score
                    )
                    if result.category_id == self.evalConfig.names.index(self.evalConfig.targetName):
                        self.dt.append(result)
                        noRelevantDet = False
                if noRelevantDet:
                    result = BBoxResult(
                        image_id=image.id, category_id=self.evalConfig.names.index(self.evalConfig.targetName),
                        bbox=None,
                        score=None
                    )
                    self.dt.append(result)

            self.dt.save(f"{self.cacheDir}/dt.json")
            cleanDt = self.dt.search(lambda r: r.bbox is not None and r.score is not None)
            print(f"{len(cleanDt)=}")
            cleanDt.save(f"{self.cacheDir}/cleanDt.json")
        else:
            raise ValueError
    
    def evaluate(self):
        if self.gt is None or self.dt is None:
            self.calc_dt()
        
        cocoGt = COCO(f'{self.cacheDir}/gt.json')
        cocoDt = cocoGt.loadRes(f'{self.cacheDir}/cleanDt.json')
        cocoeval = COCOeval(
            cocoGt=cocoGt,
            cocoDt=cocoDt,
            iouType='bbox'
        )
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()
    
    def show_eval_results(self):
        if self.gt is None or self.dt is None:
            self.calc_dt()
        s = Dataset.PreviewSettings
        s.showBBox = True
        s.showLabel = False
        s.showSeg = False
        # s.showResultLabel = True
        self.gt.show_preview(results=self.dt)

    def infer_new_annotations(
        self, newLimit: int=20,
        wcImgPaths: str='/home/clayton/workspace/prj/mediatech_poc2/main/img/*.png',
        labelmeAnnDir: str=f"/home/clayton/workspace/prj/mediatech_poc2/main/img/json_pending"
    ):
        if self.gt is None:
            self.load_gt()
        imgPaths = sorted(glob.glob(wcImgPaths))
        doneFilenames = [os.path.basename(image.file_name) for image in self.gt.images]
        imgPaths = [imgPath for imgPath in imgPaths if os.path.basename(imgPath) not in doneFilenames]
        imgPaths = imgPaths[:newLimit]

        newDataset = Dataset()
        from datetime import datetime
        import time
        newDataset.info.date_created = datetime.now()
        newDataset.info.year = datetime.now().year
        newDataset.info.description = "Inferred results. Needs editing."
        newDataset.licenses = self.gt.licenses.copy()
        newDataset.categories = self.gt.categories.copy()

        for imgPath in tqdm(imgPaths, desc="Inferring New"):
            newDt = Results()
            newDt._objects.clear()
            img = cv2.imread(imgPath)
            h, w = img.shape[:2]
            img[:int(h*0.75), :int(w*0.75), :] = (255, 255, 255)
            image = Image(
                id=len(newDataset.images),
                width=img.shape[1], height=img.shape[0],
                file_name=imgPath,
                license=0,
                date_captured=datetime.fromtimestamp(
                    time.mktime(time.gmtime(os.path.getctime(imgPath)))
                )
            )
            newDataset.images.append(image)
            outputs = self.predictor(img)
            instances = outputs["instances"].to("cpu")
            for box, score, catId in zip(
                instances._fields['pred_boxes'].tensor.tolist(),
                instances._fields['scores'].tolist(),
                instances._fields['pred_classes'].tolist()
            ):
                box = [int(val) for val in box]
                bbox = BBox2D(Vector2(*box[:2]), Vector2(*box[2:]))
                result = BBoxResult(
                    image_id=image.id, category_id=catId,
                    bbox=[bbox.v0.x, bbox.v0.y, bbox.xInterval.length, bbox.yInterval.length],
                    score=score
                )
                newDt.append(result)
            newDt = newDt.search(lambda r: r.category_id == self.evalConfig.names.index(self.evalConfig.targetName))
            newAnns = newDt.to_annotations(minPairingIoU=0.5)
            for newAnn in newAnns:
                newAnn.id = len(newDataset.annotations)
                newDataset.annotations.append(newAnn)
        
        newDataset.to_labelme(annDir=labelmeAnnDir, overwrite=False, allowNoAnn=True)

    def crop_detections(
        self, dumpDir: str,
        targets: list[ConfusionType]=[ConfusionType.FP_RESULT],
        iouThresh: float=0.5,
        overwrite: bool=False
    ):
        """
        TODO: Try using the Path class to make hierarchy of the dump folder more flexible.
        """
        if self.gt is None:
            self.load_gt()
        if self.dt is None:
            self.calc_dt()

        if os.path.isdir(dumpDir):
            if overwrite:
                rmtree(dumpDir)
            else:
                raise FileExistsError(f"Directory exists at: {dumpDir}")
        os.makedirs(dumpDir)

        maxImgId = max(self.gt.images, key=lambda image: image.id).id
        maxImgIdDigits = len(str(maxImgId))
        for image in tqdm(self.gt.images, desc='Cropping Detections', unit='image(s)'):
            meta = self.gt.get_frame_eval_meta(
                dt=self.dt, image_id=image.id, iouThresh=iouThresh
            )

            totalImages = 0
            for target in targets:
                if target == ConfusionType.TP_RESULT:
                    totalImages += len(meta.tpResultIdxList)
                elif target == ConfusionType.FP_RESULT:
                    totalImages += len(meta.fpResultIdxList)
                elif target == ConfusionType.TP_ANN:
                    totalImages += len(meta.tpAnnIdxList)
                elif target == ConfusionType.FN_ANN:
                    totalImages += len(meta.fnAnnIdxList)
                else:
                    raise ValueError
            if totalImages == 0:
                continue

            imgDirName = str(image.id)
            while len(imgDirName) < maxImgIdDigits:
                imgDirName = f"0{imgDirName}"
            imgDir = f"{dumpDir}/{imgDirName}"
            os.makedirs(imgDir)

            img = cv2.imread(image.file_name)
            for target in targets:
                if target == ConfusionType.TP_RESULT:
                    objs = meta.tpResults
                elif target == ConfusionType.FP_RESULT:
                    objs = meta.fpResults
                elif target == ConfusionType.TP_ANN:
                    objs = meta.tpAnns
                elif target == ConfusionType.FN_ANN:
                    objs = meta.fnAnns
                else:
                    raise ValueError
                
                if len(objs) == 0:
                    continue
                saveDir = f"{imgDir}/{target.name}"
                os.makedirs(saveDir)

                nDigits = len(str(len(objs)))
                for i, obj in enumerate(objs):
                    bbox = obj.bbox2d
                    croppedImg = img[
                        bbox.v0.y:bbox.v1.y, bbox.v0.x:bbox.v1.x, :
                    ].copy()
                    numStr = str(i)
                    while len(numStr) < nDigits:
                        numStr = f"0{numStr}"
                    savePath = f"{saveDir}/{numStr}.png"
                    cv2.imwrite(savePath, croppedImg)

