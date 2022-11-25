from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

target = 'waku'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
register_coco_instances(
    name=f'{target}_train',
    metadata={},
    json_file=f'../datasetDump/{target}Dataset.json',
    image_root='..'
)

cfg.DATASETS.TRAIN = (f'{target}_train',)
cfg.DATASETS.TEST = ()
cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.0002
cfg.SOLVER.BASE_LR = 0.0002

cfg.INPUT.MIN_SIZE_TRAIN = (736, 768, 800, 832, 864)
cfg.INPUT.MAX_SIZE_TRAIN = 30000
cfg.SOLVER.CHECKPOINT_PERIOD = 2000
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.OUTPUT_DIR = f"{target}_output0"

print(cfg)
# exit()
trainer = DefaultTrainer(cfg)
trainer.train()
