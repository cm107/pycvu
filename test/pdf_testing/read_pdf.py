from __future__ import annotations
import os
import fitz
from typing import Generator
from fitz.fitz import Document, Page, Pixmap, Rect
from PIL import Image
from pyevu import Vector2
from pycvu import CvUtil, Convert
import cv2

class PDF:
    def __init__(self, path: str):
        self.doc: Document = fitz.open(path)

    def __len__(self) -> int:
        return len(self.doc)

    def __iter__(self) -> Generator[PDFPage]:
        for i in range(len(self)):
            yield self[i]
    
    def __getitem__(self, idx) -> PDFPage:
        if type(idx) is int:
            return PDFPage(self.doc[idx])
        else:
            raise TypeError

class PDFPage:
    def __init__(self, _page: Page):
        self._page = _page
        initPix: Pixmap = _page.get_pixmap()
        self._initDpi: tuple[int, int] = (initPix.xres, initPix.yres)
        self._initResolution = (initPix.width, initPix.height)

        self._dpi: tuple[int, int] = self._initDpi
        self._resolution: tuple[int, int] = self._initResolution

        self._dpiScaleFactor: tuple[float, float] = (1, 1)

    @property
    def dpi(self) -> tuple[int, int]:
        return self._dpi
    
    @dpi.setter
    def dpi(self, value: tuple[int, int]):
        self._dpi = value
        self._dpiScaleFactor = tuple([
            target / init
            for init, target in zip(self._initDpi, self._dpi)
        ])
        self._resolution = tuple([
            int(scale * init)
            for scale, init in zip(self._dpiScaleFactor, self._initResolution)
        ])
    
    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution
    
    @resolution.setter
    def resolution(self, value: tuple[int, int]):
        self._resolution = value
        self._dpiScaleFactor = tuple([
            current / init
            for current, init in zip(self._resolution, self._initResolution)
        ])
        self._dpi = tuple([
            int(scale * init)
            for scale, init in zip(self._dpiScaleFactor, self._initDpi)
        ])

    @property
    def dpiScaleFactor(self) -> tuple[float, float]:
        return self._dpi
    
    @dpiScaleFactor.setter
    def dpiScaleFactor(self, value: tuple[float, float]):
        self._dpiScaleFactor = value
        self._dpi = tuple([
            int(scale * init)
            for scale, init in zip(self._dpiScaleFactor, self._initDpi)
        ])
        self._resolution = tuple([
            int(scale * init)
            for scale, init in zip(self._dpiScaleFactor, self._initResolution)
        ])
    
    @property
    def width(self) -> int:
        self._resolution[0]
    
    @width.setter
    def width(self, value: int):
        self.resolution = (value, self.height)

    @property
    def height(self) -> int:
        return self._resolution[1]
    
    @height.setter
    def height(self, value: int):
        self.resolution = (self.width, value)

    def get_pixmap(self) -> Pixmap:
        return self._page.get_pixmap(
            matrix=fitz.Matrix(*self._dpiScaleFactor)
        )
    
    def get_image(self) -> Image.Image:
        pix = self.get_pixmap()
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    def search_for(self, query: str, hit_max: int=16) -> list[Rect]:
        return self._page.search_for(query, hit_max=hit_max)

    def draw_rect(self, img: Image.Image, rects: list[Rect]) -> Image.Image:
        img = Convert.pil_to_cv(img)
        xscale, yscale = self._dpiScaleFactor
        for rect in rects:
            pt1 = Vector2(rect.x0 * xscale, rect.y0 * yscale)
            pt2 = Vector2(rect.x1 * xscale, rect.y1 * yscale)
            img = CvUtil.rectangle(
                img, pt1=pt1, pt2=pt2,
                color=(0,0,255), thickness=1, lineType=cv2.LINE_AA
            )
        img = Convert.cv_to_pil(img)
        return img
    
    def get_text(self, textType: str='text') -> list[str]:
        return self._page.get_text(textType).split('\n')

path = "/home/clayton/Desktop/35206365.pdf"
doc = PDF(path)

page = doc[0]
page.dpi = (256, 256)
print(f"{page.get_text()=}")
img = page.get_image()
img.save('/home/clayton/Desktop/35206365.png')
img = page.draw_rect(
    img=img,
    rects=page.search_for('井上', hit_max=15)
)
img.show()
