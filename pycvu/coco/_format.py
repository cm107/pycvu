from __future__ import annotations
from datetime import datetime
from ..base import Base, BaseHandler

from typing import TypeVar

class CocoBase(Base):
    def __init__(self):
        super().__init__()
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, item_dict: dict):
        return cls(**item_dict)

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
            item_dict['date_created'] = self.date_created.timestamp()
        return item_dict
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Info:
        params = item_dict.copy()
        if params['date_created'] is not None:
            params['date_created'] = datetime.fromtimestamp(params['date_created'])
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
            item_dict['date_captured'] = self.date_captured.timestamp()
        return item_dict

    @classmethod
    def from_dict(cls, item_dict: dict) -> Image:
        params = item_dict.copy()
        if params['date_captured'] is not None:
            params['date_captured'] = datetime.fromtimestamp(params['date_captured'])
        return Image(**params)

class Images(BaseHandler[Image]):
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

class Licenses(BaseHandler[License]):
    def __init__(self, _objects: list[License]=None):
        super().__init__(_objects)

    def to_dict(self) -> list[dict]:
        return [obj.to_dict() for obj in self]
    
    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Licenses:
        return Licenses([License.from_dict(val) for val in item_dict])
