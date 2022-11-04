from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Type
import cv2
import numpy as np

from pycvu.util import CvUtil, PilUtil
from ..vector import Vector
from ..color import Color, HSV
from ..interval import Interval
if TYPE_CHECKING:
    from ._artist import Artist

@classmethod
def debug(cls: Type[Artist]):
    cls.maskSetting.track = True; cls.maskSetting.canBeOccluded = True; cls.maskSetting.canOcclude = True

    colorInterval: Interval[Color] = Interval[Color](Color.black, Color.white)

    img = np.zeros((500, 500, 3), dtype=np.uint8)
    # cls.color = Color(255, 0, 0)
    cls.color = colorInterval.random()
    cls.thickness = 1

    p0 = Vector(200, 200); p1 = Vector(300, 200)
    r = int(Vector.Distance(p0, p1))
    center = 0.5 * (p0 + p1)
    width = 3 * r; height = 2 * r
    rectShape = Vector(width, height)
    c0 = center - 0.5 * rectShape
    c1 = center + 0.5 * rectShape

    drawer = cls(img)
    (
        drawer
        .circle(center=p0, radius=r)
        .circle(center=p1, radius=r)
        .line(p0, p1)
        .rectangle(c0, c1)
    )
    # cls.color = Color(0, 0, 255)
    cls.color = colorInterval.random()
    cls.thickness = 4
    offset = (Vector.down + Vector.right).normalized * 50
    (
        drawer
        .circle(center=p0 + offset, radius=r, fill=True)
        .circle(center=p1 + offset, radius=r, fill=True)
        .line(p0 + offset, p1 + offset)
        .rectangle(c0 + offset, c1 + offset)
    )
    # cls.color = Color(0, 255, 0)
    cls.color = colorInterval.random()
    offset = (Vector.down + Vector.left).normalized * 50
    (
        drawer
        .circle(center=p0 + offset, radius=r)
        .circle(center=p1 + offset, radius=r)
        .line(p0 + offset, p1 + offset)
        .rectangle(c0 + offset, c1 + offset, fill=True)
    )
    # cls.color = Color(255, 255, 0)
    cls.color = colorInterval.random()
    offset = Vector.up * 100
    drawer.ellipse(
        center=center+offset, axis=(70, 30),
        angle=30, startAngle=90, endAngle=270,
        fill=True
    )
    if True: # TODO: Need to apply this to masks as well.
        # drawer.resize(fx=1.1, fy=0.9)
        drawer.resize(fx=2, fy=0.9)
        drawer.affine_rotate(45, adjustBorder=True)
    # cls.color = Color(255, 0, 0)
    cls.color = colorInterval.random()
    cls.fontScale = 2.0
    drawer.text("Hello World!", org=(100, 100))
    drawer.text("Hello World!", org=(100, 200), bottomLeftOrigin=True)

    drawer.pil.text(text="荒唐無稽", position=(300, 300))
    
    cls.color = Color(255, 0, 0)
    cls.PIL.fontSize = 50
    cls.PIL.hankoOutlineWidth = 10
    cls.PIL.hankoMarginRatio = 0.1
    drawer.pil.hanko(text="合格", position=(300, 300+200))

    cls.color = Interval[HSV](HSV(0, 0.9375, 0.5), HSV(359.9, 1.0, 1.0))
    positionCallback = CvUtil.Callback.get_position_interval
    
    for i in range(10):
        drawer.line(pt1=positionCallback, pt2=positionCallback)
        drawer.rectangle(pt1=positionCallback, pt2=positionCallback, fill=False)
        drawer.ellipse(
            center=positionCallback,
            axis=Interval[Vector[float]](Vector[float](5, 5), Vector[float](100, 100)),
            angle=Interval[float](0, 180),
            startAngle=Interval[float](0, 360),
            endAngle=Interval[float](0, 360),
            fill=False
        )
        drawer.circle(
            center=positionCallback,
            radius=Interval[int](5, 100),
            fill=True
        )
        cls.PIL.fontSize = Interval[int](5, 40)
        cls.PIL.hankoOutlineWidth = Interval[int](1, 5)
        cls.PIL.hankoMarginRatio = Interval[float](0.1, 0.5)
        drawer.pil.hanko(text='合格', position=PilUtil.Callback.get_position_interval)

    # TODO: Serialize the mask settings.
    # drawer.save('/tmp/artistDebugSave.json', saveImg=False, saveMeta=True)
    # del drawer
    # drawer = cls.load('/tmp/artistDebugSave.json', img=img, loadMeta=True) # Make sure save and load works.

    result, maskHandler = drawer.draw_and_get_masks()

    maskHandler.show_preview()
    for mask in maskHandler:
        if mask._mask.sum() == 0:
            continue
        mask.show_preview(showBBox=True, showContours=True)

    cv2.imshow('debug', result)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
