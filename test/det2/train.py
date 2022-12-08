from __future__ import annotations
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from fvcore.common.config import CfgNode
from detectron2.data import build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper
from typing import TypeVar
from kkdetection.detectron2.mapper import Mapper

from detectron2.data.common import AspectRatioGroupedDataset

from pycvu.detectron2.frame import FrameData
from pycvu.vis.cv import SimpleVisualizer

M = TypeVar('M', bound=DatasetMapper) # Must inherit from DatasetMapper

class Trainer(DefaultTrainer):
    def __init__(self, cfg: CfgNode, mapper: M=None):
        self.mapper = mapper
        super().__init__(cfg)
    
    @property
    def data_loader(self) -> AspectRatioGroupedDataset:
        return self._trainer.data_loader

    def build_train_loader(self, cfg):
        """
        Returns:
            iterable
        """
        return build_detection_train_loader(cfg, mapper=self.mapper)
    
    def preview_augmentation(self):
        # import random
        # for data in random.sample(self.data_loader, k=5):
        for batch in self.data_loader:
            vis = SimpleVisualizer()
            with vis.loop(batch) as loop:
                while not loop.done:
                    data = batch[loop.index]
                    frame = FrameData(**data)
                    frame.check_fields()
                    img = frame.preview_image
                    vis.show(img, f'{loop.index=}')
            break
    
    def train(self):
        with open(f'{self.cfg.OUTPUT_DIR}/trainCfgSettings.txt', 'w') as f:
            f.write(str(self.cfg))
        super().train()

class TrainConfig:
    def __init__(
        self,
        image_root: str, json_file: str,
        modelZooConfig: str,
        baseLr: float, checkpointPeriod: int, maxIter: int,
        imsPerBatch: int,
        minSizeTrain: int | tuple, maxSizeTrain: int,
        names: list[str],
        outputDir: str,
        augConfigPath: str=None
    ):
        self.image_root = image_root
        self.json_file = json_file
        self.modelZooConfig = modelZooConfig

        self.baseLr = baseLr
        self.checkpointPeriod = checkpointPeriod
        self.maxIter = maxIter
        self.imsPerBatch = imsPerBatch

        self.minSizeTrain = minSizeTrain
        self.maxSizeTrain = maxSizeTrain
        self.names = names

        self.outputDir = outputDir

        self.augConfigPath = augConfigPath

    @classmethod
    @property
    def new_hanko(cls) -> TrainConfig:
        return TrainConfig(
            image_root='../datasetDump/hankoDataset.json',
            json_file='..',
            modelZooConfig="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            baseLr=0.0002, checkpointPeriod=2000, maxIter=10000,
            imsPerBatch=6,
            minSizeTrain=(736, 768, 800, 832, 864),
            maxSizeTrain=30000,
            names=['hanko'],
            outputDir="train_output/hanko_output-kumeaug-00",
            augConfigPath='/home/clayton/workspace/prj/mediatech_poc2/main/augconfig.json'
        )
    
    @classmethod
    @property
    def new_hanko_kume_dataset(cls) -> TrainConfig:
        config = TrainConfig.new_hanko
        config.image_root = '/home/clayton/workspace/prj/mediatech_poc2/main/train_hanko'
        config.json_file = '/home/clayton/workspace/prj/mediatech_poc2/main/coco_train_hanko.json'
        config.names = ['hanko', 'name']
        config.outputDir = "train_output/custom_hanko-kume_dataset-kumeaug-01"
        config.maxIter = 10000
        return config

    @classmethod
    @property
    def new_hanko_kume_dataset_cascade(cls) -> TrainConfig:
        config = TrainConfig.new_hanko_kume_dataset
        config.modelZooConfig = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
        config.imsPerBatch = 4
        config.baseLr = 0.00025
        config.outputDir = "train_output/cascade_hanko-kume_dataset-kumeaug-01"
        return config

    @classmethod
    @property
    def new_hanko_kume_dataset_long(cls) -> TrainConfig:
        config = TrainConfig.new_hanko_kume_dataset
        config.outputDir = "train_output/custom_hanko-kume_dataset-kumeaug-long-00"
        config.checkpointPeriod = 10000
        config.maxIter = 50000
        return config

    @classmethod
    @property
    def new_hanko_kume_dataset_cascade_long(cls) -> TrainConfig:
        config = TrainConfig.new_hanko_kume_dataset_cascade
        config.outputDir = "train_output/cascade_hanko-kume_dataset-kumeaug-long-00"
        config.checkpointPeriod = 20000
        config.maxIter = 100000
        return config

class TrainSession:
    def __init__(self, trainConfig: TrainConfig):
        self.trainConfig = trainConfig

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(self.trainConfig.modelZooConfig))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.trainConfig.modelZooConfig)
        register_coco_instances(
            name='train',
            metadata={},
            json_file=self.trainConfig.json_file,
            image_root=self.trainConfig.image_root
        )

        self.cfg.DATASETS.TRAIN = ('train',)
        self.cfg.DATASETS.TEST = ('train',)
        self.cfg.SOLVER.WARMUP_METHOD = "linear"
        self.cfg.SOLVER.WARMUP_FACTOR = 1.0 / self.cfg.SOLVER.WARMUP_ITERS
        self.cfg.SOLVER.BASE_LR = self.trainConfig.baseLr
        self.cfg.SOLVER.CHECKPOINT_PERIOD = self.trainConfig.checkpointPeriod
        self.cfg.SOLVER.MAX_ITER = self.trainConfig.maxIter
        self.cfg.SOLVER.IMS_PER_BATCH = self.trainConfig.imsPerBatch
        self.cfg.SOLVER.STEPS = (30000,)

        self.cfg.INPUT.RANDOM_FLIP = 'none'
        self.cfg.INPUT.MIN_SIZE_TRAIN = self.trainConfig.minSizeTrain
        self.cfg.INPUT.MAX_SIZE_TRAIN = self.trainConfig.maxSizeTrain

        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.trainConfig.names)
        self.cfg.OUTPUT_DIR = self.trainConfig.outputDir

        self.cfg.MODEL.MASK_ON     = False
        self.cfg.MODEL.KEYPOINT_ON = False
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 0

        self.cfg.VIS_PERIOD = 100
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 200

        if self.trainConfig.augConfigPath is not None:
            mapper = Mapper(self.cfg, config=self.trainConfig.augConfigPath)
        else:
            mapper = None
        self.trainer = Trainer(self.cfg, mapper=mapper)

session = TrainSession(TrainConfig.new_hanko_kume_dataset_cascade)
# session.trainer.preview_augmentation()
session.trainer.train()
