from typing import TypeVar
from .._format import CocoBase, Info, Images, Licenses, \
    AnnotationsBase, CategoriesBase

class DatasetBase(CocoBase):
    @property
    def _info(self) -> Info:
        assert hasattr(self, 'info')
        return getattr(self, 'info')
    
    @_info.setter
    def _info(self, val: Info):
        assert hasattr(self, 'info')
        setattr(self, 'info', val)

    @property
    def _images(self) -> Images:
        assert hasattr(self, 'images')
        return getattr(self, 'images')
    
    @_images.setter
    def _images(self, val: Images):
        assert hasattr(self, 'images')
        setattr(self, 'images', val)

    @property
    def _licenses(self) -> Licenses:
        assert hasattr(self, 'licenses')
        return getattr(self, 'licenses')
    
    @_licenses.setter
    def _licenses(self, val: Licenses):
        assert hasattr(self, 'licenses')
        setattr(self, 'licenses', val)

    @property
    def _annotations(self) -> AnnotationsBase:
        assert hasattr(self, 'annotations')
        return getattr(self, 'annotations')
    
    @_annotations.setter
    def _annotations(self, val):
        assert hasattr(self, 'annotations')
        setattr(self, 'annotations', val)

    @property
    def _categories(self) -> CategoriesBase:
        assert hasattr(self, 'categories')
        return getattr(self, 'categories')
    
    @_categories.setter
    def _categories(self, val):
        assert hasattr(self, 'categories')
        setattr(self, 'categories', val)

    from ._manipulation import combine, filter, reindex, split

DS = TypeVar('DS', bound=DatasetBase)