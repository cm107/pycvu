from __future__ import annotations
from typing import Callable
from datetime import datetime
import copy
from tqdm import tqdm
from ..base import Base, BaseHandler

from typing import TypeVar

class CocoBase(Base):
    def __init__(self):
        super().__init__()
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, item_dict: dict):
        constr_params = cls.get_constructor_params()
        return cls(**{key: val for key, val in item_dict.items() if key in constr_params})

T = TypeVar('T', bound=CocoBase)
class CocoBaseHandler(BaseHandler[T]):
    def __init__(self, _objects: list[T]=None):
        super().__init__(_objects)
    
    def reindex(
        self, showPbar: bool=False, leavePbar: bool=False,
        applyToSelf: bool=True,
        idUpdateCallback: Callable[[int, int],]=None
    ):
        if not applyToSelf:
            result = copy.deepcopy(self)
        else:
            result = self
        result.sort(key=lambda obj: obj.id)

        if showPbar:
            pbar = tqdm(total=len(result), leave=leavePbar)
            pbar.set_description(f"Reindexing {type(self).__name__}")
        for i, obj in enumerate(result):
            if idUpdateCallback is not None:
                idUpdateCallback(obj.id, i)
            obj.id = i
            if showPbar:
                pbar.update()
        if showPbar:
            pbar.close()
        return result

class Info(CocoBase):
    def __init__(
        self,
        year: int=None, version: str=None, description: str=None,
        contributor: str=None, url: str=None, date_created: datetime=None
    ):
        self.year = year
        self.version = version
        self.description = description
        self.contributor = contributor
        self.url = url
        self.date_created = date_created

    def to_dict(self) -> dict:
        item_dict = self.__dict__.copy()
        if self.date_created is not None:
            if type(self.date_created) is datetime:
                item_dict['date_created'] = self.date_created.timestamp()
        return item_dict
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Info:
        params = item_dict.copy()
        if params['date_created'] is not None:
            if type(params['date_created']) is int \
                or (
                    type(params['date_created']) is float
                    and params['date_created'] % 1 == 0
                ):
                params['date_created'] = datetime.fromtimestamp(params['date_created'])
            else:
                params['date_created'] = None
        return Info(**params)

class Image(CocoBase):
    def __init__(
        self, id: int, width: int, height: int, file_name: str,
        license: int=None, flickr_url: str=None, coco_url: str=None, date_captured: datetime=None
    ):
        self.id = id; self.width = width; self.height = height
        self.file_name = file_name
        """Assume that this can be either just the filename or an entire path."""
        self.license = license; self.flickr_url = flickr_url; self.coco_url = coco_url
        self.date_captured = date_captured

    def to_dict(self) -> dict:
        item_dict = self.__dict__.copy()
        if item_dict['date_captured'] is not None:
            if type(item_dict['date_captured']) is int \
                or (
                    type(item_dict['date_captured']) is float
                    and item_dict['date_captured'] % 1 == 0
                ):
                item_dict['date_captured'] = self.date_captured.timestamp()
            else:
                item_dict['date_captured'] = None
        return item_dict

    @classmethod
    def from_dict(cls, item_dict: dict) -> Image:
        params = item_dict.copy()
        if params['date_captured'] is not None:
            if type(params['date_captured']) is int:
                params['date_captured'] = datetime.fromtimestamp(params['date_captured'])
        return Image(**params)

class Images(CocoBaseHandler[Image]):
    def __init__(self, _objects: list[Image]=None):
        super().__init__(_objects)

    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Images:
        return Images([Image.from_dict(val) for val in item_dict])

class License(CocoBase):
    def __init__(self, id: int, name: str, url: str):
        self.id = id; self.name = name; self.url = url

class Licenses(CocoBaseHandler[License]):
    def __init__(self, _objects: list[License]=None):
        super().__init__(_objects)

    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Licenses:
        return Licenses([License.from_dict(val) for val in item_dict])
