from __future__ import annotations
import json
import os

class Base:
    def __init__(self):
        pass

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, item_dict: dict):
        cls(**item_dict)

    def save(self, path: str):
        json.dump(self.to_dict(), open(path, 'w'))

    @classmethod
    def load(cls, path: str):
        return cls.from_dict(json.load(open(path, 'r')))

class Dataset(Base):
    def __init__(self, frames: list[Frame]):
        self.frames = frames
    
    def to_dict(self):
        return [frame.to_dict() for frame in self.frames]

    def from_dict(cls, item_dict: dict) -> Dataset:
        return Dataset([Frame.load(_dict) for _dict in item_dict])

    @classmethod
    def from_dir(cls, dirPath: str) -> Dataset:
        if not os.path.isdir(dirPath):
            raise FileNotFoundError
        annPath = f"{dirPath}/annotations.json"
        if not os.path.isfile(annPath):
            raise FileNotFoundError
        return cls.load(annPath)

class Frame(Base):
    def __init__(self, imagefilename: str, annotation: list[Annotation]):
        self.imagefilename = imagefilename
        """Image filename"""
        self.annotation = annotation
        """List of object block annotations"""
    
    def to_dict(self) -> dict:
        return {
            "imagefilename": self.imagefilename,
            "annotation": [ann.to_dict() for ann in self.annotation]
        }

    @classmethod
    def from_dict(cls, item_dict: dict) -> Frame:
        return Frame(
            imagefilename=item_dict['imagefilename'],
            annotation=[Annotation.from_dict(_dict) for _dict in item_dict['annotation']]
        )

class Annotation(Base):
    def __init__(self, coordinates: Coordinates, label: str):
        self.coordinates = coordinates
        """Coordinates of object block"""
        self.label = label
        """Label of object"""
    
    def to_dict(self) -> dict:
        return {
            "coordinates": self.coordinates.to_dict(),
            "label": self.label
        }
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> Annotation:
        return Annotation(
            coordinates=Coordinates.from_dict(item_dict['coordinates']),
            label=item_dict['label']
        )

class Coordinates(Base):
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        """Center x coordinate of block"""
        self.y = y
        """Center y coordinate of block"""
        self.width = width
        """Width of block"""
        self.height = height
        """Height of block"""
