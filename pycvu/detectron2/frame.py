from __future__ import annotations
from typing import Any
import torch
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import numpy as np
import numpy.typing as npt
import cv2
from pyevu import BBox2D, Vector2

from ..util import CvUtil, ColorVar, Convert, PilUtil
from ..base import Base, BaseHandler
from ..color import Color
from datumaro.util.mask_tools import mask_to_rle, rle_to_mask

class FrameData:
    def __init__(
        self,
        file_name: str,
        height: int, width: int,
        image_id: int, image: torch.Tensor[torch.uint8],
        instances: Instances
    ):
        self.file_name = file_name
        self.height = height; self.width = width
        self.image_id = image_id; self.image = image
        self.instances = instances

        self.converted = FrameData.ConvertedProperties(self)
    
    @property
    def fields(self) -> dict[str, Any]:
        return self.instances._fields

    @property
    def gt_boxes(self) -> Boxes:
        return self.fields['gt_boxes'] if 'gt_boxes' in self.fields else None

    @property
    def gt_classes(self) -> torch.Tensor:
        return self.fields['gt_classes'] if 'gt_classes' in self.fields else None
    
    def check_fields(self):
        keys: list[str] = []
        typeList: list[type] = []
        for key, val in self.fields.items():
            if key not in ['gt_boxes', 'gt_classes']:
                keys.append(key)
                typeList.append(type(val))
        if len(keys) > 0:
            msg = f"TODO: Implement property for fields:"
            for key, typeVal in zip(keys, typeList):
                msg += f"\n\tfields['{key}'] ({typeVal})"
            raise Exception(msg)
    
    @property
    def preview_image(self) -> npt.NDArray[np.uint8]:
        img: np.ndarray = self.converted.image.copy()
        
        bboxList = self.converted.gt_boxes
        for bbox in bboxList:
            img = CvUtil.rectangle(
                img=img,
                pt1=bbox.v0, pt2=bbox.v1,
                color=(0,0,255), thickness=1,
                lineType=cv2.LINE_AA
            )
        return img

    class ConvertedProperties:
        def __init__(self, _data: FrameData):
            self._data = _data

        @property
        def image(self) -> npt.NDArray[np.uint8]:
            return self._data.image.permute(1, 2, 0).numpy()

        @property
        def gt_boxes(self) -> list[BBox2D]:
            if self._data.gt_boxes is not None:
                return [
                    BBox2D(Vector2(*vals[:2]), Vector2(*vals[2:]))
                    for vals in self._data.gt_boxes.tensor.tolist()
                ]
            else:
                return None
        
        @property
        def gt_classes(self) -> list[int]:
            if self._data.gt_classes is not None:
                return self._data.gt_classes.tolist()
            else:
                return None

class FrameInstance(Base):
    class Draw:
        color: ColorVar = Color.red
        thickness: int = 1
        lineType: int = cv2.LINE_AA
        textSizeProp: float = 0.25
        showScore: bool = True
        showLabel: bool = False

    def __init__(
        self,
        pred_box: BBox2D,
        pred_class: int,
        score: float,
        pred_mask: dict=None
    ):
        self.pred_box = pred_box
        self.pred_class = pred_class
        self.score = score
        self.pred_mask = pred_mask

    def to_dict(self) -> dict:
        item_dict = {
            'pred_box': self.pred_box.to_dict(),
            'pred_class': self.pred_class,
            'score': self.score
        }
        if self.pred_mask is not None:
            item_dict['pred_mask'] = self.pred_mask
            item_dict['pred_mask']['counts'] = item_dict['pred_mask']['counts'].tolist()
        return item_dict

    @classmethod
    def from_dict(cls, item_dict: dict) -> FrameInstance:
        if 'pred_mask' in item_dict:
            pred_mask = item_dict['pred_mask']
            pred_mask['counts'] = np.array(pred_mask['counts'], dtype=np.uint32)
        else:
            pred_mask = None
        return FrameInstance(
            pred_box=BBox2D.from_dict(item_dict['pred_box']),
            pred_class=item_dict['pred_class'],
            score=item_dict['score'],
            pred_mask=pred_mask
        )
    
    @property
    def bool_mask(self) -> np.ndarray:
        if self.pred_mask is None:
            return None
        return rle_to_mask(self.pred_mask).astype(np.bool_)

    def _draw(self, img: np.ndarray, labels: list[str]=None) -> np.ndarray:
        # img = CvUtil.rectangle(
        #     img=img,
        #     pt1=self.pred_box.v0, pt2=self.pred_box.v1,
        #     color=FrameInstance.Draw.color,
        #     thickness=FrameInstance.Draw.thickness,
        #     lineType=FrameInstance.Draw.lineType
        # )
        # img = CvUtil.bbox_text(
        #     img, text=str(round(self.score, 2)),
        #     bbox=self.pred_box, color=FrameInstance.Draw.color
        # )
        img = Convert.cv_to_pil(img)

        textParts: list[str] = []
        scoreText = str(round(self.score, 2))
        if FrameInstance.Draw.showScore:
            textParts.append(scoreText)
        if labels is not None:
            labelText = labels[self.pred_class]
            if FrameInstance.Draw.showLabel:
                textParts.append(labelText)

        text = ' '.join(textParts)
        img = PilUtil.bbox_text(
            img=img, text=text, bbox=self.pred_box,
            fontPath=PilUtil.defaultFontPath,
            color=FrameInstance.Draw.color,
            side='right',
            direction='rtl',
            targetProp=FrameInstance.Draw.textSizeProp, targetDim='height',
            drawBbox=True
        )
        img = Convert.pil_to_cv(img)
        return img
    
    def draw(self, img: np.ndarray, labels: list[str]=None) -> np.ndarray:
        result = img.copy()
        result = self._draw(result, labels=labels)
        return result

class FrameInstances(BaseHandler[FrameInstance]):
    def __init__(
        self,
        _objects: list[FrameInstance]=None,
        _image_size: tuple[int, int]=None
    ):
        super().__init__(_objects)
        self._image_size = _image_size
    
    def to_dict(self) -> dict:
        return {
            '_objects': [obj.to_dict() for obj in self],
            '_image_size': list(self._image_size) if self._image_size is not None else None
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> FrameInstances:
        return FrameInstances(
            [FrameInstance.from_dict(_dict) for _dict in item_dict['_objects']],
            _image_size=tuple(item_dict['_image_size']) if item_dict['_image_size'] is not None else None
        )

    @property
    def image_height(self) -> int:
        return self._image_size[0] if self._image_size is not None else None
    
    @property
    def image_width(self) -> int:
        return self._image_size[1] if self._image_size is not None else None

    @property
    def num_instances(self) -> int:
        return self.__len__()

    @classmethod
    def from_raw_instances(cls, instances: Instances) -> FrameInstances:
        fields: dict[str, Any] = instances._fields
        # image_height, image_width = instances._image_size
        
        pred_boxes: Boxes = fields['pred_boxes'] if 'pred_boxes' in fields else None
        if pred_boxes is not None:
            pred_boxes: list[BBox2D] = [
                BBox2D(Vector2(*vals[:2]), Vector2(*vals[2:]))
                for vals in pred_boxes.tensor.tolist()
            ]
        pred_masks: torch.Tensor = fields['pred_masks'] if 'pred_masks' in fields else None
        if pred_masks is not None:
            pred_masks = [np.array(mask, dtype=np.bool_) for mask in pred_masks.tolist()]
            pred_masks = [mask_to_rle(mask) for mask in pred_masks]
        else:
            pred_masks = [None] * len(pred_boxes)

        pred_classes: torch.Tensor = fields['pred_classes'] if 'pred_classes' in fields else None
        pred_classes: list[int] = pred_classes.tolist()
        scores: torch.Tensor = fields['scores'] if 'scores' in fields else None
        scores: list[float] = [float(score) for score in scores.tolist()]

        # num_instances = len(pred_boxes) # *
        assert pred_boxes is not None
        assert pred_classes is not None
        assert scores is not None
        return FrameInstances(
            [
                FrameInstance(pred_box, pred_class, score, pred_mask)
                for pred_box, pred_class, score, pred_mask in zip(
                    pred_boxes, pred_classes, scores, pred_masks
                )
            ],
            _image_size=instances._image_size
        )
    
    def _draw(self, img: np.ndarray, labels: list[str]=None) -> np.ndarray:
        for obj in self:
            img = obj._draw(img, labels=labels)
        return img
    
    def draw(self, img: np.ndarray, labels: list[str]=None) -> np.ndarray:
        result = img.copy()
        result = self._draw(result, labels=labels)
        return result