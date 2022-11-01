from __future__ import annotations
from typing import TYPE_CHECKING, Type
import cv2
import numpy as np
from ..vector import Vector
from ..color import Color
from ..interval import Interval
if TYPE_CHECKING:
    from ._artist import Artist

@classmethod
def debug(cls: Type[Artist]):
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
    drawer.resize(fx=1.1, fy=0.9)
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

    # print(f"{[type(callback) for callback in drawer._drawQueue]}")
    # import functools
    # p: functools.partial = drawer._drawQueue[0]
    
    # print("dir(p)")
    # for key in dir(p):
    #     print(f"\tgetattr(p, {key}): {getattr(p, key)}")

    drawer.save('/tmp/artistDebugSave.json', saveImg=False, saveMeta=True)
    del drawer
    drawer = cls.load('/tmp/artistDebugSave.json', img=img, loadMeta=True) # Make sure save and load works.

    result = drawer.draw()

    cv2.imshow('debug', result)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
