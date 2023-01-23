from __future__ import annotations
from typing import Callable
import numpy as np
from ._parsed import Base, Drawings, Resolution
from ....base import BaseHandler
from pycvu.util import Convert, CvUtil
from PIL import Image
import cv2

class LineMeta(Base):
    def __init__(self, line: Drawings.Line, res: Resolution):
        self.line = line
        self.res = res
    
    def to_dict(self) -> dict:
        return {
            'line': self.line.to_dict(),
            'res': self.res.to_dict()
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> LineMeta:
        return LineMeta(
            line=Drawings.Line.from_dict(item_dict['line']),
            res=Resolution.from_dict(item_dict['res'])
        )

class LineMatch(Base):
    def __init__(self, lineMeta: LineMeta, hits: list[str]):
        super().__init__()
        self.lineMeta = lineMeta
        self.hits = hits
    
    def to_dict(self) -> dict:
        return {
            'lineMeta': self.lineMeta.to_dict(),
            'hits': self.hits
        }

    @classmethod
    def from_dict(cls, item_dict: dict) -> LineMatch:
        return LineMatch(
            lineMeta=LineMeta.from_dict(item_dict['lineMeta']),
            hits=item_dict['hits']
        )

class LineMatches(BaseHandler[LineMatch]):
    def __init__(self, _objects: list[LineMatch]=None):
        super().__init__(_objects)
    
    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> LineMatches:
        return LineMatches([LineMatch.from_dict(_dict) for _dict in item_dict])

from typing import TypeVar
from pyevu import Grid2D, Vector2

T = TypeVar('T', bound=Base)

class MetaGrid(Grid2D[T]):
    def __init__(self, res: Resolution, width: int=4, height: int=4):
        self.res = res
        super().__init__(
            width=width, height=height,
            cellWidth=res.width / width, cellHeight=res.height / height,
            origin=Vector2.zero, countCorners=False
        )

class LineMetaCell(Base):
    def __init__(self, lines: list[Drawings.Line]=None, hitMap: list[list[str]]=None):
        self.lines = lines if lines is not None else []
        self.hitMap = hitMap if hitMap is not None else []
    
    def to_dict(self) -> dict:
        return {
            'lines': [line.to_dict() for line in self.lines],
            'hitMap': self.hitMap
        }

    @classmethod
    def from_dict(cls, item_dict: dict) -> LineMetaCell:
        return LineMetaCell(
            lines=[Drawings.Line.from_dict(_dict) for _dict in item_dict['lines']],
            hitMap=item_dict['hitMap']
        )

    @property
    def paths(self) -> list[str]:
        _paths: list[str] = []
        for paths in self.hitMap:
            for path in paths:
                if path not in _paths:
                    _paths.append(path)
        _paths.sort()
        return _paths

    def add(self, line: Drawings.Line, path: str):
        if line not in self.lines:
            self.lines.append(line)
            self.hitMap.append([path])
        else:
            self.hitMap[self.lines.index(line)].append(path)
    
    def contains(self, line: Drawings.Line) -> bool:
        return line in self.lines
    
    def prune(
        self,
        includeLines: list[Drawings.Line]=None,
        includePaths: list[str]=None,
        excludeLines: list[Drawings.Line]=None,
        excludePaths: list[str]=None
    ):
        # I think it's okay to make this an in-place operation.
        for i in list(range(len(self.lines)))[::-1]:
            if includeLines is not None:
                if self.lines[i] not in includeLines:
                    del self.lines[i]
                    del self.hitMap[i]
            if includePaths is not None:
                for j in list(range(len(self.hitMap[i])))[::-1]:
                    if self.hitMap[i][j] not in includePaths:
                        del self.hitMap[i][j]
                if len(self.hitMap[i]) == 0:
                    del self.lines[i]
                    del self.hitMap[i]
            if excludeLines is not None:
                if self.lines[i] in excludeLines:
                    del self.lines[i]
                    del self.hitMap[i]
            if excludePaths is not None:
                for j in list(range(len(self.hitMap[i])))[::-1]:
                    if self.hitMap[i][j] in excludePaths:
                        del self.hitMap[i][j]
                if len(self.hitMap[i]) == 0:
                    del self.lines[i]
                    del self.hitMap[i]

    @property
    def lineRanking(self) -> list[dict]:
        ranking = [
            {'line': self.lines[idx], 'hits': hits}
            for idx, hits in enumerate(self.hitMap)
        ]
        ranking.sort(key=lambda rank: len(rank['hits']), reverse=True)
        return ranking

class LineMetaGrid(MetaGrid[LineMetaCell]):
    def __init__(
        self, res: Resolution, width: int=4, height: int=4,
        _obj_arr: np.ndarray=None
    ):
        super().__init__(
            res=res, width=width, height=height
        )

        if _obj_arr is not None:
            self._obj_arr = _obj_arr
        else:
            self.clear()
    
    def to_dict(self) -> dict:
        return {
            'res': self.res.to_dict(),
            'width': self.width,
            'height': self.height,
            'data': [
                cell.to_dict()
                for cell in self._obj_arr.flat
            ]
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> LineMetaGrid:
        width = item_dict['width']
        height = item_dict['height']

        data = [
            LineMetaCell.from_dict(_dict)
            for _dict in item_dict['data']
        ]
        _obj_arr = np.array(data).reshape(width, height)

        return LineMetaGrid(
            res=Resolution.from_dict(item_dict['res']),
            width=width,
            height=height,
            _obj_arr=_obj_arr
        )

    @property
    def paths(self) -> list[str]:
        paths: list[str] = []
        
        def func(grid: LineMetaGrid, c: Vector2):
            for path in grid[c].paths:
                if path not in paths:
                    paths.append(path)
        
        self.LoopCoords(func)
        return paths

    def add(self, line: Drawings.Line, path: str):
        c = self.World2GridCoord(line.p0)
        if self.gridBoundingBox.Contains(c):
            self[c].add(line=line, path=path)
    
    def contains(self, line: Drawings.Line) -> bool:
        c = self.World2GridCoord(line.p0)
        return self[c].contains(line)

    def prune(
        self,
        includeLines: list[Drawings.Line]=None,
        includePaths: list[str]=None,
        excludeLines: list[Drawings.Line]=None,
        excludePaths: list[str]=None
    ):
        # I think it's okay to make this an in-place operation.
        self.LoopCoords(
            lambda grid, c: grid[c].prune(
                includeLines=includeLines,
                includePaths=includePaths,
                excludeLines=excludeLines,
                excludePaths=excludePaths
            )
        )
    
    def clear(self):
        def initGrid(grid: LineMetaGrid, c: Vector2):
            grid[c] = LineMetaCell()
        self.LoopCoords(initGrid)

    @property
    def lineRanking(self) -> list[dict]:
        ranking: list[dict] = []

        def process(grid: LineMetaGrid, c: Vector2):
            ranking.extend(
                grid[c].lineRanking
            )
        
        self.LoopCoords(process)
        ranking.sort(key=lambda rank: len(rank['hits']), reverse=True)
        return ranking

class LineMetaGridHandler(BaseHandler[LineMetaGrid]):
    def __init__(
        self,
        width: int=4, height: int=4,
        _objects: list[LineMetaGridHandler]=None
    ):
        self.width = width
        self.height = height
        super().__init__(_objects)
    
    def add(self, line: Drawings.Line, path: str, res: Resolution):
        grid = self.get(lambda grid: grid.res == res)
        if grid is not None:
            grid.add(line=line, path=path)
        else:
            grid = LineMetaGrid(res=res, width=self.width, height=self.height)
            grid.add(line=line, path=path)
            self.append(grid)
    
    def to_dict(self) -> dict:
        return {
            'width': self.width,
            'height': self.height,
            'grids': [obj.to_dict() for obj in self._objects]
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> LineMetaGridHandler:
        return LineMetaGridHandler(
            width=item_dict['width'],
            height=item_dict['height'],
            _objects=[LineMetaGrid.from_dict(_dict) for _dict in item_dict['grids']]
        )

class LineFormat(Base):
    def __init__(
        self, name: str, res: Resolution,
        lines: list[Drawings.Line],
        paths: list[str]=None
    ):
        self.name = name
        self.res = res
        self.lines = lines
        self.paths = paths if paths is not None else []

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'res': self.res.to_dict(),
            'lines': [line.to_dict() for line in self.lines],
            'paths': self.paths
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> LineFormat:
        return LineFormat(
            name=item_dict['name'],
            res=Resolution.from_dict(item_dict['res']),
            lines=[Drawings.Line.from_dict(_dict) for _dict in item_dict['lines']],
            paths=item_dict['paths']
        )
    
    def matches(self, lines: list[Drawings.Line]) -> bool:
        for line in self.lines:
            if line not in lines:
                return False
        return True
    
    def get_preview(self) -> Image.Image:
        img = Image.new("RGB", (self.res.width, self.res.height), color=(0, 0, 0))
        img = Convert.pil_to_cv(img)
        for line in self.lines:
            img = CvUtil.line(
                img, pt1=line.p0, pt2=line.p1,
                color=(255,255,255), thickness=1, lineType=cv2.LINE_AA
            )
        img = Convert.cv_to_pil(img)
        return img
    
    def save_preview(self, path: str):
        img = self.get_preview()
        img.save(path)

class LineFormats(BaseHandler[LineFormat]):
    def __init__(self, _objects: list[LineFormat]=None):
        super().__init__(_objects)
    
    def to_dict(self) -> dict:
        return [obj.to_dict() for obj in self]
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> LineFormats:
        return LineFormats([LineFormat.from_dict(_dict) for _dict in item_dict])
