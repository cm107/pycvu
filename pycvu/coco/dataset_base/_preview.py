from __future__ import annotations
import os
import cv2
import numpy as np
from tqdm import tqdm
from ...vis.cv import SimpleVisualizer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..dataset_base._base import DS
    from .._format import Image

__all__ = [
    'draw_preview',
    'get_preview_from_image',
    'get_preview_from_image_idx',
    'get_preview',
    'show_preview',
    'show_filename',
    'save_preview',
    'save_filename',
]

def draw_preview(self: DS, img: np.ndarray, image_id: int, results=None) -> np.ndarray:
    """Override this method"""
    raise NotImplementedError("draw_preview method hasn't been overridden")

def get_preview_from_image(
    self: DS, image: Image, results=None,
    imgDir: str=None
) -> np.ndarray:
    if imgDir is not None:
        if not os.path.isdir(imgDir):
            raise FileNotFoundError
        path = f"{imgDir}/{image.file_name}"
    else:
        path = image.file_name
    if not os.path.isfile(path):
        raise FileNotFoundError
    img = cv2.imread(path)
    assert img is not None
    img = self.draw_preview(
        img=img, image_id=image.id,
        results=results
    )
    return img

def get_preview_from_image_idx(
    self: DS, imageIdx: int, results=None,
    imgDir: str=None
) -> np.ndarray:
    image = self.images[imageIdx]
    return self.get_preview_from_image(
        image=image, results=results, imgDir=imgDir
    )

def get_preview(
    self: DS, image: Image | int, results=None,
    imgDir: str=None
) -> np.ndarray:
    if type(image) is int:
        image = self.images[image]
    return self.get_preview_from_image(
        image=image, results=results, imgDir=imgDir
    )

def show_preview(
    self: DS, results=None,
    imgDir: str=None
):
    vis = SimpleVisualizer()
    with vis.loop(self.images) as loop:
        while not loop.done:
            image = self.images[loop.index]
            img = self.get_preview_from_image_idx(loop.index, results=results, imgDir=imgDir)
            vis.show(img, title=f'image.id={image.id}, filename={os.path.basename(image.file_name)}')

def show_filename(
    self: DS, filename: str,
    results=None, imgDir: str=None
):
    image = self.images.get(lambda img: os.path.basename(img.file_name) == filename)
    if image is None:
        raise FileNotFoundError(f"Failed to find image in dataset with a filename of {filename}")
    img = self.get_preview_from_image(image, results=results, imgDir=imgDir)
    vis = SimpleVisualizer()
    vis.show(img, title=f"filename={os.path.basename(image.file_name)}")

def save_preview(
    self: DS, saveDir: str,
    results=None, imgDir: str=None, showPbar: bool=False
):
    os.makedirs(saveDir, exist_ok=True)
    if showPbar:
        pbar = tqdm(total=len(self.images), leave=False)
    for i, image in enumerate(self.images):
        img = self.get_preview_from_image_idx(i, results=results, imgDir=imgDir)
        filename = os.path.basename(image.file_name)
        savePath = f"{saveDir}/{filename}"
        cv2.imwrite(savePath, img)
        if showPbar:
            pbar.update()
    if showPbar:
        pbar.close()

def save_filename(
    self: DS, filename: str, savePath: str,
    results=None, imgDir: str=None
):
    image = self.images.get(lambda img: os.path.basename(img.file_name) == filename)
    if image is None:
        raise FileNotFoundError(f"Failed to find image in dataset with a filename of {filename}")
    img = self.get_preview_from_image(image, results=results, imgDir=imgDir)
    cv2.imwrite(savePath, img)
