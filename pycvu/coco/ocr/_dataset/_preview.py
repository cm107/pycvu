from __future__ import annotations
import cv2
import numpy as np
from pyevu import BBox2D, Vector2
from ....polygon import Segmentation
from ....util import PilUtil, Convert
from ....color import Color

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Dataset

from ...dataset_base._preview import *

class PreviewSettings:
    showGT: bool = True
    showResult: bool = False

    showText: bool = True
    textColor: Color = Color.red
    textSide: str = 'top'
    textTargetProp: float = 0.8
    textTargetDim: str = 'width'

    showQuad: bool = True
    quadColor: Color = Color.red
    quadThickness: int = 2

def draw_preview(self: Dataset, img: np.ndarray, image_id: int, results=None) -> np.ndarray:
    img = Convert.cv_to_pil(img)
    s = PreviewSettings
    if s.showGT:
        anns = self.annotations.search(
            lambda ann: ann.image_id == image_id)
        for ann in anns:
            text = ann.text
            quad = ann.quad2d

            if text is not None and PreviewSettings.showText:
                img = PilUtil.bbox_text(
                    img=img, text=text, bbox=quad.bbox2d,
                    fontPath=PilUtil.defaultFontPath, color=PreviewSettings.textColor,
                    side=PreviewSettings.textSide,
                    targetProp=PreviewSettings.textTargetProp,
                    targetDim=PreviewSettings.textTargetDim
                )
            if quad is not None and PreviewSettings.showQuad:
                for line in quad.lines:
                    img = PilUtil.line(
                        img=img, pt1=line.p0, pt2=line.p1,
                        color=PreviewSettings.quadColor,
                        thickness=PreviewSettings.quadThickness
                    )
        
    if s.showResult:
        raise NotImplementedError
    return Convert.pil_to_cv(img)
