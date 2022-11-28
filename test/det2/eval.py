from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from pycvu.coco.object_detection import *
from pycvu.vis.cv import SimpleVisualizer
import random
import cv2
import os
import glob
from pyevu import Vector2, BBox2D

target = 'hanko'

cfg = get_cfg()
if False:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = f"{target}_output0/model_final.pth"
    # cfg.MODEL.WEIGHTS = "waku_output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

    cfg.INPUT.MIN_SIZE_TEST = 1600
    cfg.INPUT.MAX_SIZE_TEST = 5000

    predictor = DefaultPredictor(cfg)

    # register_coco_instances(
    #     name=f'{target}_train',
    #     metadata={},
    #     json_file=f'../datasetDump/{target}Dataset.json',
    #     image_root='..'
    # )
    # metadata = MetadataCatalog.get(f"{target}_train")
else:
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "/home/clayton/workspace/prj/data/MediaTech/250_終了時点ファイル/weight_20220715_iter10000_hanko_name/model_final.pth"
    cfg.INPUT.MIN_SIZE_TEST = 1600
    cfg.INPUT.MAX_SIZE_TEST = 5000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)

gt = Dataset.from_labelme('/home/clayton/workspace/prj/mediatech_poc2/main/img/json', allowNoAnn=True)
gt.filter(catFilter=lambda cat: cat.name == target, reindex=True, applyToSelf=True, cleanupImg=False)
dt = Results()
print(f"{gt.categories=}")

for image in gt.images:
    img = cv2.imread(image.file_name)
    h, w = img.shape[:2]
    img[:int(h*0.75), :int(w*0.75), :] = (255, 255, 255)
    outputs = predictor(img)
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
        dt.append(result)

dt = dt.search(lambda r: r.category_id == 0)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt.save('/tmp/gt.json')
dt.save('/tmp/dt.json')
dt.to_annotations(minPairingIoU=0.5).save('/tmp/dt_ann.json')

gt.show_preview(results=dt)

cocoGt = COCO('/tmp/gt.json')
cocoDt = cocoGt.loadRes('/tmp/dt.json')
cocoeval = COCOeval(
    cocoGt=cocoGt,
    cocoDt=cocoDt,
    iouType='bbox'
)
cocoeval.evaluate()
cocoeval.accumulate()
cocoeval.summarize()

# Infer new annotations
inferNewAnn = True
newLimit = 20
if inferNewAnn:
    imgDir = '/home/clayton/workspace/prj/mediatech_poc2/main/img'
    imgPaths = sorted(glob.glob(f"{imgDir}/*.png"))
    doneFilenames = [os.path.basename(image.file_name) for image in gt.images]
    imgPaths = [imgPath for imgPath in imgPaths if os.path.basename(imgPath) not in doneFilenames]
    imgPaths = imgPaths[:newLimit]

    newDataset = Dataset()
    from datetime import datetime
    import time
    newDataset.info.date_created = datetime.now()
    newDataset.info.year = datetime.now().year
    newDataset.info.description = "Inferred results. Needs editing."
    newDataset.licenses = gt.licenses.copy()
    newDataset.categories = gt.categories.copy()

    for imgPath in imgPaths:
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
        outputs = predictor(img)
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
        newDt = newDt.search(lambda r: r.category_id == 0)
        newAnns = newDt.to_annotations(minPairingIoU=0.5)
        for newAnn in newAnns:
            newAnn.id = len(newDataset.annotations)
            newDataset.annotations.append(newAnn)
    
    
    newDataset.to_labelme(annDir=f"{imgDir}/json_pending", overwrite=False, allowNoAnn=True)
