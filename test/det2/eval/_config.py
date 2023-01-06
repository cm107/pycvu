from __future__ import annotations
from pycvu.base import Base

"""
Overall Notes 12/8/2022:
    I am evaluating the trained models using about 100 images that I annotated myself.
    
    So far I am unable to train a model to the same performance as Kume's
    using the small dataset that I generated in the mediatech folder.
    I think that maybe the size of the dataset is insufficient.
    I need to try re-generating a larger dataset and train off of that.

    I created a custom dataset for training in addition to the dataset that was generated
    by Kume's mediatech repo. Combining this custom dataset with Kume's dataset seems to
    result in a slight increase in performance.

    So far most of my testing has been focused on the hanko detections.
    Name and waku models doen't seem to perform very well, even using the weights that Kume trained.

    TODO:
    " Try regenerating a larger version of Kume's dataset and train with that.
    * Try including distractor objects in the custom dataset.
    * Try training waku model.
"""

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
    def kume_name(self) -> EvalConfig:
        return EvalConfig(
            modelZooConfig="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            weights="/home/clayton/workspace/prj/data/MediaTech/250_終了時点ファイル/weight_20220715_iter10000_hanko_name/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['hanko', 'name'], targetName='name',
            gtLabelmeDir='/home/clayton/workspace/prj/mediatech_poc2/main/img/json'
        )

    @classmethod
    @property
    def kume_waku(self) -> EvalConfig:
        return EvalConfig(
            modelZooConfig="Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            weights="/home/clayton/workspace/prj/data/MediaTech/250_終了時点ファイル/weight_20220711_iter25000/model_final.pth",
            minSizeTest=1600, maxSizeTest=5000,
            scoreThreshTest=0.7,
            names=['waku'], targetName='waku',
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
