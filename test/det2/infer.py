from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from pycvu.vis.cv import SimpleVisualizer
import random
import cv2
import glob

"""
TODO:   For now, set up a simple evaluation script.
        I need to set up an evaluation that can help show the weaknesses of the current model.
        This should help me decide what changes to make during the next training session.
        By the looks of it right now, it seems like I may have to introduce some image augmentations to pick up the cases that aren't being detected?
"""

target = 'hanko'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = f"{target}_output0/model_final.pth"
# cfg.MODEL.WEIGHTS = "waku_output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

cfg.INPUT.MIN_SIZE_TEST = 1600
cfg.INPUT.MAX_SIZE_TEST = 5000

predictor = DefaultPredictor(cfg)

register_coco_instances(
    name=f'{target}_train',
    metadata={},
    json_file=f'../datasetDump/{target}Dataset.json',
    image_root='..'
)
metadata = MetadataCatalog.get(f"{target}_train")

vis = SimpleVisualizer()
# dataset = Dataset.load(metadata.json_file)
img_dir = "/home/clayton/workspace/prj/mediatech_poc2/main/test"
imgPaths = glob.glob(f"{img_dir}/*.png")
# with vis.loop(random.sample(dataset.images._objects, 20)) as loop:
with vis.loop(random.sample(imgPaths, 20)) as loop:
    while not loop.done:
        # image = loop._iter[loop.index]
        # path = f"{metadata.image_root}/{image.file_name}"
        path = loop._iter[loop.index]
        img = cv2.imread(path)
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(instances)
        vis.show(out.get_image()[:, :, ::-1], title="infer")
