from __future__ import annotations
import numpy as np
import cv2
from pyevu.quad2 import Quad2
from ..base import Base, BaseHandler
from ..util import CvUtil
from ..color import Color

class OcrResult(Base):
    class Draw:
        color = Color.red
        thickness = 1
        lineType = cv2.LINE_AA

    def __init__(self, quad: Quad2, text: str, score: float):
        self.quad = quad
        self.text = text
        self.score = score
    
    def to_dict(self) -> dict:
        return {
            'quad': self.quad.to_list(),
            'text': self.text,
            'score': self.score
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> OcrResult:
        return OcrResult(
            quad=Quad2.from_list(item_dict['quad']),
            text=item_dict['text'],
            score=item_dict['score']
        )
    
    @classmethod
    def from_easyocr(cls, data: tuple[list, str, float]) -> OcrResult:
        return OcrResult(
            Quad2.from_list(data[0]),
            text=data[1],
            score=data[2]
        )

    def _draw(self, img: np.ndarray):
        for line in self.quad.lines:
            img = CvUtil.line(
                img, pt1=line.p0, pt2=line.p1,
                color=OcrResult.Draw.color,
                thickness=OcrResult.Draw.thickness,
                lineType=OcrResult.Draw.lineType
            )
    
    def draw(self, img: np.ndarray) -> np.ndarray:
        result = img.copy()
        self._draw(result)
        return result

class OcrResults(BaseHandler[OcrResult]):
    def __init__(self, _objects: list[OcrResult]=None):
        super().__init__(_objects)
    
    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> OcrResults:
        return OcrResults(
            [
                OcrResult.from_dict(_dict)
                for _dict in item_dict
            ]
        )
    
    @classmethod
    def from_easyocr(cls, data: list[tuple[list, str, float]]) -> OcrResults:
        return OcrResults([OcrResult.from_easyocr(_data) for _data in data])

    def _draw(self, img: np.ndarray):
        for obj in self:
            obj._draw(img)
    
    def draw(self, img: np.ndarray) -> np.ndarray:
        result = img.copy()
        self._draw(result)
        return result
