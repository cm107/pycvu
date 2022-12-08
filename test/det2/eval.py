from __future__ import annotations
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from pycvu.coco.object_detection import *
from pycvu.vis.cv import SimpleVisualizer
from pycvu.base import Base
import random
import cv2
import os
import glob
from tqdm import tqdm
from pyevu import Vector2, BBox2D
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from enum import Enum
from datetime import datetime

class EvalConfig(Base):
    def __init__(
        self,
        modelZooConfig: str, weights: str,
        minSizeTest: int, maxSizeTest: int,
        scoreThreshTest: float,
        names: list[str],
        targetName: str, gtLabelmeDir: str
    ):
        self.modelZooConfig = modelZooConfig
        self.weights = weights
        self.minSizeTest = minSizeTest
        self.maxSizeTest = maxSizeTest
        self.scoreThreshTest = scoreThreshTest
        self.names = names
        
        self.targetName = targetName
        self.gtLabelmeDir = gtLabelmeDir

    @classmethod
    @property
    def kume_hanko(self) -> EvalConfig:
        """
        Trained by kume with Kume's dataset using Misc/cascade_mask_rcnn_R_50_FPN_3x.
        Performs well.
        """
        return EvalConfig(
            modelZooConfig="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            weights="/home/clayton/workspace/prj/data/MediaTech/250_終了時点ファイル/weight_20220715_iter10000_hanko_name/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko', 'name'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def kume_hanko_retrain(self) -> EvalConfig:
        """
        Retrained by Clayton with Kume's dataset using Misc/cascade_mask_rcnn_R_50_FPN_3x.
        Doesn't perform as well as kume_hanko, but still quite well.
        Could probably perform better with more training.
        """
        return EvalConfig(
            modelZooConfig="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            weights="/home/clayton/workspace/prj/mediatech_poc2/main/output_hanko/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko', 'name'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def new_hanko(self) -> EvalConfig:
        """
        Trained by Clayton with custom dataset using COCO-Detection/faster_rcnn_R_50_FPN_3x.
        Using same augmentation settings as Kume's model.
        Possible causes of poor performance:
        * Dataset is bad.
        * COCO-Detection/faster_rcnn_R_50_FPN_3x isn't good enough.
        * Need to train longer?
        * After looking at the augmentation preview, it looks like the hanko being occluded by other shapes may be causing problems.

        TODO: Try training with Kume's dataset but with COCO-Detection/faster_rcnn_R_50_FPN_3x.
        TODO: Try getting rid of occlusion on hanko data in my dataset and train again.
        """
        return EvalConfig(
            modelZooConfig="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            weights=f"train_output/hanko_output-kumeaug-00/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def new_hanko_kume_dataset(self) -> EvalConfig:
        """
        Seems to detect the hanko most of the time, but there are a lot of false positives
        with high scores. This could be due to insufficient training, or it could be a problem
        with the model itself.
        TODO: Train with Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml with the same number of iterations.
              If the evaluation results don't change considerably, the likely reason for the difference
              in performance between this model and the one that Kume trained is simply insufficient training time.
              Otherwise, Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml should be utilized from now on.
        """
        return EvalConfig(
            modelZooConfig="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            weights=f"train_output/custom_hanko-kume_dataset-kumeaug-01/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko', 'name'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def new_hanko_kume_dataset_cascade(self) -> EvalConfig:
        """
        The performance seems even worse than before.
        That likely means that the training time is too short.
        """
        return EvalConfig(
            modelZooConfig="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            weights="train_output/cascade_hanko-kume_dataset-kumeaug-01/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko', 'name'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def new_hanko_kume_dataset_long(self) -> EvalConfig:
        """Better than before, but still too many false positives."""
        return EvalConfig(
            modelZooConfig="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            weights=f"train_output/custom_hanko-kume_dataset-kumeaug-long-00/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko', 'name'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def new_hanko_kume_dataset_cascade_long(self) -> EvalConfig:
        """
        Much better than before, but not as good as the one that Kume trained.
        It is still picking up objects that look similar to a hanko, but not as much as before.
        Kume's model was trained in 4.5 hours. There must be some difference in the training settings.
        """
        return EvalConfig(
            modelZooConfig="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            weights="train_output/cascade_hanko-kume_dataset-kumeaug-long-00/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko', 'name'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def kk_custom_hanko_cascade(self) -> EvalConfig:
        """
        Trained using a custom dataset, but with kume's repository.
        The results are much better than what was yielded by this respostory, even with the
        exact same detectron2 cfg.
        There must be a difference somewhere else in the repository that accounts for the
        better performance.

        Note: This is a hanko-only dataset.
        The performance still isn't as good as the model that kume trained, which was trained
        with both hanko and name annotations at the same time.

        Next: Try training with both hanko and name annotations and see if that helps the model's
              overall performance.
        """
        return EvalConfig(
            modelZooConfig="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            weights="/home/clayton/workspace/prj/mediatech_poc2/main/custom_hanko_output-00/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def kk_custom_hanko_name_cascade(self) -> EvalConfig:
        """
        Slightly better than kk_custom_hanko_cascade, but not by much.
        """
        return EvalConfig(
            modelZooConfig="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            weights="/home/clayton/workspace/prj/mediatech_poc2/main/custom_hanko-name_output-00/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko', 'name'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def kk_mixed_hanko_name_cascade(self) -> EvalConfig:
        """
        Mixed custom dataset with kume's dataset. Sampled such that both are 50/50.
        Mixing the two kinds of datasets seems to have helped.
        The performance is quite high.
        (Although not as high as the model that kume trained. Maybe he trained on a different dataset.)

        Next: Sample some distractor objects for the next dataset.
        """
        return EvalConfig(
            modelZooConfig="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            weights="/home/clayton/workspace/prj/mediatech_poc2/main/mixed_hanko-name_output-00/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko', 'name'], targetName='hanko',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

class CacheMode(Enum):
    READONLY = 0
    NEW = 1
    UPDATE = 2

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

# session = EvalSession(EvalConfig.kume_hanko)
# session = EvalSession(EvalConfig.kume_hanko_retrain)
# session = EvalSession(EvalConfig.kk_custom_hanko_cascade)
# session = EvalSession(EvalConfig.kk_custom_hanko_name_cascade)
session = EvalSession(EvalConfig.kk_mixed_hanko_name_cascade)
session.evaluate()
session.show_eval_results()
# session.infer_new_annotations(newLimit=20)
