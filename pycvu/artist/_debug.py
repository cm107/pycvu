from __future__ import annotations
from typing import TYPE_CHECKING, Type
import cv2
import numpy as np
from pyevu import Vector2
if TYPE_CHECKING:
    from ._artist import Artist

@classmethod
def debug(cls: Type[Artist]):
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cls.color = (0, 0, 255)
    cls.thickness = 1

    p0 = Vector2(200, 200); p1 = Vector2(300, 200)
    r = int(Vector2.Distance(p0, p1))
    center = 0.5 * (p0 + p1)
    width = 3 * r; height = 2 * r
    rectShape = Vector2(width, height)
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
    cls.color = (255, 0, 0)
    cls.thickness = 4
    offset = (Vector2.down + Vector2.right).normalized * 50
    (
        drawer
        .circle(center=p0 + offset, radius=r, fill=True)
        .circle(center=p1 + offset, radius=r, fill=True)
        .line(p0 + offset, p1 + offset)
        .rectangle(c0 + offset, c1 + offset)
    )
    cls.color = (0, 255, 0)
    offset = (Vector2.down + Vector2.left).normalized * 50
    (
        drawer
        .circle(center=p0 + offset, radius=r)
        .circle(center=p1 + offset, radius=r)
        .line(p0 + offset, p1 + offset)
        .rectangle(c0 + offset, c1 + offset, fill=True)
    )
    cls.color = (255, 255, 0)
    offset = Vector2.up * 100
    drawer.ellipse(
        center=center+offset, axis=(70, 30),
        angle=30, startAngle=90, endAngle=270,
        fill=True
    )
    drawer.resize(fx=1.1, fy=0.9)
    drawer.affine_rotate(45, adjustBorder=True)
    cls.color = (0, 0, 255)
    cls.fontScale = 2.0
    drawer.text("Hello World!", org=(100, 100))
    drawer.text("Hello World!", org=(100, 200), bottomLeftOrigin=True)
    
    drawer.pil.text(text="荒唐無稽", position=(300, 300))
    
    cls.color = (255, 0, 0)
    cls.PIL.fontSize = 50
    cls.PIL.hankoOutlineWidth = 10
    cls.PIL.hankoMarginRatio = 0.1
    drawer.pil.hanko(text="合格", position=(300, 300+200))

    result = drawer.draw()
    cv2.imshow('debug', result)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
