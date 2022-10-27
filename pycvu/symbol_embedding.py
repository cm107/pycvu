from __future__ import annotations
from functools import partial
import glob
from inspect import signature
import json
import random
from typing import Callable
import cv2
import os

import numpy as np
from .base import Base, BaseHandler
from kkannotation.symbolemb import SymbolEmbedding as KumeEmbedding
from kkannotation.coco import CocoManager
from kkannotation.util.image import mask_from_bool_to_polygon

"""
Note: Based on Kume's implementation.
"""

class Canvas(Base):
    def __init__(self, type: str, path: str, width: int, height: int):
        super().__init__()
        self.type = type
        self.path = path
        self.width = width
        self.height = height
    
    def __call__(self) -> np.ndarray:
        if self.type == "noise":
            params: list[str] = list(signature(KumeEmbedding.create_canvas).parameters.keys())
            return KumeEmbedding.create_canvas(**{key: val for key, val in self.to_dict().items() if key in params})
        else:
            backgrounds = glob.glob(self.path)
            p = partial(KumeEmbedding.create_canvas_from_image, path=backgrounds)
            params: list[str] = list(signature(p).parameters.keys())
            params.remove('path')
            return p(**{key: val for key, val in self.to_dict().items() if key in params})

    @classmethod
    @property
    def example(self) -> Canvas:
        return Canvas(type="background", path="./background/*.png", width=800, height=800)

class Label(Base):
    def __init__(self, name: str, path: str, thre: list[int]):
        super().__init__()
        self.name = name
        self.path = path
        self.thre = thre

        self.img: np.ndarray = None
        self.mask: np.ndarray = None
        self.load_img()

    def load_img(self):
        paths = glob.glob(self.path)
        assert len(paths) > 0
        path = paths[0]
        self.img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:,:,:3]
        self.mask = KumeEmbedding.get_mask(self.img, self.thre)

    @classmethod
    @property
    def example(self) -> Label:
        return Label(name="label00", path="./symbol/A1_21_label0_1.png", thre=[0, 220])

class Labels(BaseHandler[Label]):
    def __init__(self, _objects: list[Label]=None):
        super().__init__(_objects)
    
    @classmethod
    @property
    def example(self) -> Labels:
        return Labels([Label.example] * 3)

class Proc(Base):
    """
    Kume, why did you design your config like this???
    I guess he wanted to create a config for each of his drawing methods, which take different parameters.
    If he was going to do this, he should have just made a config for each drawing type and saved us from this headache...
    """
    def __init__(
        self, type: str, iters: int,
        group: list[str]=None,
        chars: str=None,
        n_connect: int=None,
        range_scale: list[int]=None, range_thickness: list[int]=None,
        range_color: list[int]=None, range_rotation: list[int]=None,
        range_radius: list[int]=None, range_noise: list[int]=None,
        n_merge: int=None, n_lines: int=None,
        is_PIL: bool=None, is_hanko: bool=None, is_fix_scale_ratio: bool=None,
        font_pil: str=None # Is this a str? What is this?
    ):
        super().__init__()
        self.type = type
        self.iters = iters
        self.group = group
        self.chars = chars
        self.n_connect = n_connect
        self.range_scale = range_scale
        self.range_thickness = range_thickness
        self.range_color = range_color
        self.range_rotation = range_rotation
        self.range_radius = range_radius
        self.range_noise = range_noise
        self.n_merge = n_merge
        self.n_lines = n_lines
        self.is_PIL = is_PIL
        self.is_hanko = is_hanko
        self.is_fix_scale_ratio = is_fix_scale_ratio
        self.font_pil = font_pil

    @classmethod
    @property
    def example(self) -> Proc:
        return Proc(
            type="text", iters=50,
            chars="1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&()-=^~|@`[{;+:*]},<.>/?_",
            range_scale=[0.4, 0.5],
            range_thickness=[1, 2],
            range_color=[0, 50],
            range_rotation=[0, 0],
            n_merge=10,
            is_PIL=False,
            font_pil=None
        )

class Procs(BaseHandler[Proc]):
    def __init__(self, _objects: list[Proc]=None):
        super().__init__(_objects)
    
    @classmethod
    @property
    def example(self) -> Procs:
        return Procs([Proc.example] * 3)

    def apply(self, emb: SymbolEmbedding, filepath: str, is_save: bool=True) -> tuple[np.ndarray, CocoManager]:
        # I really don't like this... Kume's implementation is very sloppy.
        # This was made to be a lot more complicated and unorganized than it needed to be.

        draw_procs: list[Callable[[],list[np.ndarray]]] = []
        label_procs: list[Callable[[],tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        dest_procs: list[Callable[[],tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        
        for proc in self:
            def fill_params(p: partial[list]) -> partial[list]:
                params: list[str] = list(signature(p).parameters.keys())
                p = partial(p, **{key: val for key, val in emb.__dict__.items() if key in params})
                p = partial(p, **{key: val for key, val in emb.canvas.to_dict().items() if key in params})
                p = partial(p, **{key: val for key, val in proc.to_dict().items() if key in params and val is not None})
                return p

            # No img input for any of these? 💦 I really want to implement this differently.
            # This is highly unmodular. Not scalable and hard to use.
            if proc.type == 'text':
                p = partial(KumeEmbedding.draw_text)
                p = fill_params(p)
                draw_procs.append(p)
            elif proc.type == 'line':
                p = partial(KumeEmbedding.draw_shape_line)
                p = fill_params(p)
                draw_procs.append(p)
            elif proc.type == 'circle':
                p = partial(KumeEmbedding.draw_shape_circle)
                p = fill_params(p)
                draw_procs.append(p)
            elif proc.type == 'ellipse':
                p = partial(KumeEmbedding.draw_shape_ellipse)
                p = fill_params(p)
                draw_procs.append(p)
            elif proc.type == 'rectangle':
                p = partial(KumeEmbedding.draw_shape_rectangle)
                p = fill_params(p)
                draw_procs.append(p)
            elif proc.type in ['label', 'dest']:
                labels = emb.labels.search(lambda label: label.name in proc.group)
                p = partial(
                    KumeEmbedding.draw_label,
                    imgs_label=[label.img for label in labels],
                    imgs_mask=[label.mask for label in labels],
                    imgs_name=np.array([label.name for label in labels])
                )
                p = fill_params(p)
                if proc.type == 'label':
                    label_procs.append(p)
                else:
                    dest_procs.append(p)
            else:
                raise ValueError
        
        img = emb.canvas()
        coco = CocoManager()
        adds: np.ndarray = np.stack(np.sum([p() for p in draw_procs], axis=0)).astype(np.uint8)
        mask: np.ndarray = KumeEmbedding.get_mask_except_color(img=adds, color=emb.color_init)
        for p in dest_procs:
            dest, dest_mask, _ = p()
            adds = np.concatenate([adds, dest], axis=0).astype(np.uint8)
            mask = np.concatenate([mask, dest_mask], axis=0).astype(bool)
        
        # TODO: Implement n variable and create a for loop around the random sample
        mask_duplication = np.zeros(tuple(img.shape[:2])).astype(bool)
        for p in random.sample(label_procs, k=len(label_procs)):
            labels_img, labels_mask, labels_name = p()
            for label_img, label_mask, label_name in zip(labels_img, labels_mask, labels_name):
                if mask_duplication[label_mask].sum() > 0: continue
                mask_duplication[label_mask] = True
                ndf_y, ndf_x = np.where(label_mask)
                if len(ndf_y) > 0 and len(ndf_x) > 0:
                    bbox = [int(ndf_x.min()), int(ndf_y.min()), int(ndf_x.max()-ndf_x.min()), int(ndf_y.max()-ndf_y.min())]
                    coco.add(
                        filepath, img.shape[0], img.shape[1], bbox,
                        label_name, segmentations=mask_from_bool_to_polygon(label_mask.astype(np.uint8))
                    )
                    adds = np.concatenate([adds, label_img. reshape(1, *label_img. shape)], axis=0)
                    mask = np.concatenate([mask, label_mask.reshape(1, *label_mask.shape)], axis=0)
        indexes = np.random.permutation(np.arange(adds.shape[0]))
        adds    = adds[indexes]
        mask    = mask[indexes]
        for i, add in enumerate(adds):
            img[mask[i]] = add[mask[i]]
        coco.concat_added()
        if is_save:
            cv2.imwrite(filepath, img)
            coco.save(filepath + ".json")
        return img, coco

class SymbolEmbedding(Base):
    def __init__(
        self,
        color_init: tuple[int, int, int],
        canvas: Canvas, labels: Labels, procs: Procs
    ):
        super().__init__()
        self.color_init = color_init
        self.canvas = canvas
        self.labels = labels
        self.procs = procs
    
    @classmethod
    @property
    def example(self) -> SymbolEmbedding:
        return SymbolEmbedding(
            color_init=[255, 255, 255],
            canvas=Canvas.example,
            labels=Labels.example,
            procs=Procs.example
        )
    
    @classmethod
    def from_dict(cls, item_dict: dict) -> SymbolEmbedding:
        if '_module' in item_dict and '_qualname' in item_dict:
            return super().from_dict(item_dict)
        else:
            return SymbolEmbedding(
                color_init=item_dict['color_init'],
                canvas=Canvas(**item_dict['canvas']),
                labels=Labels([Label(**labelDict) for labelDict in item_dict['labels']]),
                procs=Procs([Proc(**procDict) for procDict in item_dict['procs']])
            )

    @staticmethod
    def debug():
        dump_dir = 'dump'
        os.makedirs(dump_dir, exist_ok=True)

        # print(json.dumps(SymbolEmbedding.example.to_dict(), ensure_ascii=False, indent=2, sort_keys=False))
        emb = SymbolEmbedding.load("./config_all.json")
        # print(json.dumps(emb.to_dict(), ensure_ascii=False, indent=2, sort_keys=False))
        
        import time
        startTime = time.time()
        for i in range(3):
            img, coco = emb.procs.apply(emb, filepath=f"{dump_dir}/{i}.png", is_save=True)
            # cv2.imshow('test', img)
            # cv2.waitKey(3000)
            # cv2.destroyAllWindows()
        endTime = time.time()
        print(f"time elapsed: {endTime - startTime}")

    @staticmethod
    def kume_debug():
        dump_dir = 'kume_dump'
        os.makedirs(dump_dir, exist_ok=True)

        emb = KumeEmbedding(**json.load(open('./config_all.json', 'r')))
        import time
        startTime = time.time()
        for i in range(3):
            filename = f"{dump_dir}/{i}.png"
            img, coco = emb.create_image(filename, is_save=True)
            # cv2.imshow('test', img)
            # cv2.waitKey(3000)
            # cv2.destroyAllWindows()
        endTime = time.time()
        print(f"time elapsed: {endTime - startTime}")