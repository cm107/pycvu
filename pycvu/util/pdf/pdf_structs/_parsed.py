from __future__ import annotations
from pyevu import Vector2, Vector3, BBox2D, Line2
import math

class Base:
    def __str__(self) -> str:
        paramStr = ','.join([f"{key}={val}" for key, val in self.__dict__.items()])
        return f"{type(self).__name__}({paramStr})"

    def __repr__(self) -> str:
        return self.__str__()

    def __key(self) -> tuple:
        return tuple([self.__class__] + list(self.__dict__.values()))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__key() == other.__key()
        return NotImplemented

class Resolution(Base):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    
    def to_dict(self) -> dict:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Resolution:
        return Resolution(**item_dict)

class Drawings:
    def __init__(
        self,
        lines: list[Drawings.Line]=None,
        curves: list[Drawings.Curve]=None,
        rectangles: list[Drawings.Rect]=None,
        quads: list[Drawings.Quad]=None
    ):
        self.lines = lines if lines is not None else []
        self.curves = curves if curves is not None else []
        self.rectangles = rectangles if rectangles is not None else []
        self.quads = quads if quads is not None else []

    class Line(Line2):
        def __init__(self, p0: Vector2, p1: Vector2):
            super().__init__(p0=p0, p1=p1)
        
        def to_dict(self) -> dict:
            return {
                'p0': list(self.p0),
                'p1': list(self.p1)
            }
        
        @classmethod
        def from_dict(cls, item_dict: dict) -> Drawings.Line:
            return Drawings.Line(
                p0=Vector2(*item_dict['p0']) if item_dict['p0'] is not None else None,
                p1=Vector2(*item_dict['p1']) if item_dict['p1'] is not None else None
            )
        
        def reverse_points(self) -> Drawings.Line:
            return Drawings.Line(self.p1, self.p0)

    class Curve(Base):
        def __init__(self, points: list[Vector2]):
            self.points = points

        def to_lines(self) -> list[Drawings.Line]:
            lines: list[Drawings.Line] = []
            for i in range(1, len(self.points)):
                lines.append(Drawings.Line(self.points[i-1], self.points[i]))
            return lines

    class Rect(Base):
        def __init__(self, p0: Vector2, p1: Vector2):
            self.p0 = p0
            self.p1 = p1

        def to_bbox2d(self) -> BBox2D:
            return BBox2D(v0=self.p0, v1=self.p1)
        
        @classmethod
        def from_bbox2d(self, bbox: BBox2D) -> Drawings.Rect:
            return Drawings.Rect(p0=bbox.v0, p1=bbox.v1)
    
    class Quad(Base):
        def __init__(
            self, p0: Vector2, p1: Vector2, p2: Vector2, p3: Vector2
        ):
            self.p0 = p0
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3

        def to_lines(self) -> list[Drawings.Line]:
            lines: list[Drawings.Line] = []
            points = [self.p0, self.p1, self.p2, self.p3, self.p0]
            for i in range(1, len(points)):
                lines.append(Drawings.Line(points[i-1], points[i]))
            return lines
