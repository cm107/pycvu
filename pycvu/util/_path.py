from __future__ import annotations
import os
from typing import TypeVar
from ..base import Base

class PathPart(Base):
    def __init__(self, src):
        self.src = src
    
    def __str__(self) -> str:
        return self._part

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: PP | str) -> Path:
        if isinstance(other, PathPart) or type(other) is str:
            return Path([self, other])
        else:
            raise TypeError
    
    def __radd__(self, other: PP | str) -> Path:
        if isinstance(other, PathPart) or type(other) is str:
            return Path([other, self])
        else:
            raise TypeError

    @property
    def _part(self) -> str:
        if type(self.src) is str:
            return self.src
        else:
            raise TypeError

PP = TypeVar('PP', bound=PathPart)

class PathIntPart(PathPart):
    def __init__(self, num: int, maxNumDigits: int=None):
        super().__init__(src=num)
        self.maxNumDigits = maxNumDigits
    
    @property
    def _part(self) -> str:
        partStr = str(self.src)
        if self.maxNumDigits is None:
            return partStr
        else:
            if len(partStr) > self.maxNumDigits:
                raise Exception(f'Encountered {len(partStr)} digits. Expected up to {self.maxNumDigits}.')
            while len(partStr) < self.maxNumDigits:
                partStr = f"0{partStr}"
            return partStr

class PathExtensionPart(PathPart):
    def __init__(self, extension: str):
        super().__init__(src=extension)
    
    @property
    def _part(self) -> str:
        if self.src[0] != '.':
            return f'.{self.src}'
        else:
            return self.src

class Path(Base):
    def __init__(self, src):
        self.src = src
    
    def __str__(self) -> str:
        return self._path
    
    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: P | PP | str) -> Path:
        if isinstance(other, Path) or isinstance(other, PathPart) or type(other) is str:
            return Path([self, other])
        else:
            raise TypeError

    def __radd__(self, other: P | PP | str) -> Path:
        if isinstance(other, Path) or isinstance(other, PathPart) or type(other) is str:
            return Path([other, self])
        else:
            raise TypeError

    @property
    def _path(self) -> str:
        if type(self.src) is str:
            return self.src
        elif type(self.src) is list:
            pathStrVals: list[str] = []
            for val in self.src:
                if type(val) is str:
                    pathStrVals.append(val)
                elif isinstance(val, PathPart):
                    pathStrVals.append(val._part)
                elif isinstance(val, Path):
                    pathStrVals.append(val._path)
                else:
                    raise TypeError
            return '/'.join(pathStrVals)
        else:
            raise TypeError

    def evaluate(self, depth: int=0):
        assert type(depth) is int
        if depth == 0:
            self.src = self._path
        elif depth > 0 and type(self.src) is list:
            for val in self.src:
                if isinstance(val, Path):
                    val.evaluate(depth=depth-1)

    @property
    def isDirectory(self) -> bool:
        return os.path.isdir(self._path)
    
    @property
    def isFile(self) -> bool:
        return os.path.isfile(self._path)
    
    @property
    def parentDir(self) -> Path | None:
        _parentDir = os.path.dirname(self._path)
        if len(_parentDir) > 0:
            return Path(_parentDir)
        else:
            return None

    def makedirs(self, exist_ok: bool=True):
        os.makedirs(self._path, exist_ok=exist_ok)
    
    @staticmethod
    def debug():
        print(Path([
            '/path/to/rootDir',
            PathIntPart(12, maxNumDigits=4),
            Filename(PathIntPart(32, maxNumDigits=3), PathExtensionPart('png'))
        ]))

class Filename(Path):
    def __init__(self, basename: PP | P | str, extension: PathExtensionPart):
        super().__init__([basename])
        self.extension = extension
    
    @property
    def _path(self) -> str:
        return super()._path + self.extension._part

P = TypeVar('P', bound=Path)
