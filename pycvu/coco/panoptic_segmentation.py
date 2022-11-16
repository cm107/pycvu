from __future__ import annotations
from ..base import Base, BaseHandler
from ._format import Info, Image, Images, License, Licenses

class Annotation:
    def __init__(self):
        raise NotImplementedError

class SegmentInfo:
    def __init__(self):
        raise NotImplementedError

class Categories:
    def __init__(self):
        raise NotImplementedError
