from __future__ import annotations
from ..._format import CocoBase, Info, Images, Licenses
from ...dataset_base import DatasetBase
from .._structs import Annotations, Categories

class Dataset(DatasetBase):
    def __init__(
        self, info: Info = None, images: Images = None, licenses: Licenses = None,
        annotations: Annotations = None,
        categories: Categories = None
    ):
        self.info = info if info is not None else Info()
        self.images = images if images is not None else Images()
        self.licenses = licenses if licenses is not None else Licenses()
        self.annotations = annotations if annotations is not None else Annotations()
        self.categories = categories if categories is not None else Categories()

    def to_dict(self) -> dict:
        return {key: val.to_dict() for key, val in self.__dict__.items()}

    @classmethod
    def from_dict(cls, item_dict: list[dict]) -> Dataset:
        return Dataset(
            info=Info.from_dict(item_dict['info']),
            images=Images.from_dict(item_dict['images']),
            licenses=Licenses.from_dict(item_dict['licenses']),
            annotations=Annotations.from_dict(item_dict['annotations']),
            categories=Categories.from_dict(item_dict['categories'])
        )

    from ._labelme import to_labelme, from_labelme
    from ._createml import to_createml, from_createml
    from ._eval import FrameEvalMeta, get_frame_eval_meta
    from ._preview import PreviewSettings, draw_preview, get_preview_from_image, get_preview_from_image_idx, get_preview, \
        show_preview, show_filename, save_preview, save_filename
