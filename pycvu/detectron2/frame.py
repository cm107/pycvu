from __future__ import annotations
from typing import Any
import torch
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import numpy as np
import numpy.typing as npt
import cv2
from pyevu import BBox2D, Vector2

from ..util._cv import CvUtil

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
