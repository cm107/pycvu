from typing import TypeVar
from .._format import CocoBase, Info, Images, Licenses, \
    AnnotationsBase, CategoriesBase

class DatasetBase(CocoBase):
    @property
    def _info(self) -> Info:
        assert hasattr(self, 'info')
        return getattr(self, 'info')
    
    @property
    def _images(self) -> Images:
        assert hasattr(self, 'images')
        return getattr(self, 'images')
    
    @property
    def _licenses(self) -> Licenses:
        assert hasattr(self, 'licenses')
        return getattr(self, 'licenses')
    
    @property
    def _annotations(self) -> AnnotationsBase:
        assert hasattr(self, 'annotations')
        return getattr(self, 'annotations')
    
    @property
    def _categories(self) -> CategoriesBase:
        assert hasattr(self, 'categories')
        return getattr(self, 'categories')
    
    from ._manipulation import combine, filter, reindex

DS = TypeVar('DS', bound=DatasetBase)