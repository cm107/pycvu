from __future__ import annotations
import os
import fitz
from typing import Generator, TypeVar
from fitz.fitz import Document, Page, Pixmap, Rect, Point, Quad
from PIL import Image
from pyevu import Vector2, BBox2D
from pycvu import CvUtil, PilUtil, Convert
from pycvu.util import ColorVar
import cv2
import glob
import traceback
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

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
    
    @staticmethod
    def _pdf_to_png(
        path: str,
        output_dir: str='pdf_images', dpi: int=256,
        skip_existing: bool=True
    ):
        ext: str = 'png'
        pdf = PDF(path)
        nDigits = len(str(len(pdf) - 1))
        os.makedirs(output_dir, exist_ok=True)
        if skip_existing:
            existingSaved: list[str] = glob.glob(f"{os.path.basename(path)}.*.{ext}")
            if len(existingSaved) == len(pdf):
                return
        for i, page in enumerate(pdf):
            pageStr = str(i)
            if len(pageStr) < nDigits:
                pageStr = f"0{pageStr}"
            filename = f"{os.path.basename(path)}.{pageStr}.{ext}"
            savePath = f"{output_dir}/{filename}"
            try:
                page.dpi = (dpi, dpi)
                img = page.get_image()
                img.save(savePath)
            except Exception as e:
                with open(f"{output_dir}/{os.path.basename(path)}.{pageStr}.txt", 'w') as f:
                    tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
                    f.write(tb_str)

    @staticmethod
    def pdf_to_png(
        path: str,
        output_dir: str='pdf_images', dpi: int=256,
        skip_existing: bool=True,
        pool: int=None, save_cpus: int=2, showPbar: bool=True
    ):
        _func: partial = partial(
            PDF._pdf_to_png,
            output_dir=output_dir,
            dpi=dpi,
            skip_existing=skip_existing
        )
        if os.path.isdir(path):
            paths = glob.glob(f"{path}/*.pdf")
            if pool is not None:
                numCpus = mp.cpu_count()
                allocatedCpus = min(pool, numCpus - save_cpus)
                p = mp.Pool(allocatedCpus)
                print(f"Allocating {allocatedCpus}/{numCpus} CPUs")
                if showPbar:
                    pbar = tqdm(total=len(paths))

                def update(*a):
                    if showPbar:
                        pbar.update()
                
                for _path in paths:
                    p.apply_async(_func, args=(_path,), callback=update)
                p.close()
                p.join()
                if showPbar:
                    pbar.close()

            else:
                if showPbar:
                    pbar = tqdm(total=len(paths))
                    pbar.set_description("Converting PDF to PNG")
                for _path in paths:
                    _func(_path)
                    if showPbar:
                        pbar.update()
                if showPbar:
                    pbar.close()
        elif os.path.isfile(path):
            _func(path)
        else:
            raise FileNotFoundError(f"Failed to file file/directory: {path}")

T = TypeVar('T')

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
    
    def search_for(
        self, query: str, hit_max: int=16,
        rotate: bool=True, scale: bool=True
    ) -> list[Drawings.Rect]:
        rects: list[Rect] = self._page.search_for(query, hit_max=hit_max)
        if rects is None:
            return []
        if rotate:
            rects = [
                rect * self._page.rotation_matrix
                for rect in rects
            ]
        if scale:
            xscale, yscale = self._dpiScaleFactor
            return [
                Drawings.Rect(
                    Vector2(rect.x0 * xscale, rect.y0 * yscale),
                    Vector2(rect.x1 * xscale, rect.y1 * yscale)
                )
                for rect in rects
            ]
        else:
            return [
                Drawings.Rect(
                    Vector2(rect.x0, rect.y0),
                    Vector2(rect.x1, rect.y1)
                )
                for rect in rects
            ]

    def scale_shape(self, shape: T) -> T:
        xscale, yscale = self._dpiScaleFactor
        if type(shape) is Drawings.Line:
            return Drawings.Line(
                p0=Vector2(shape.p0.x * xscale, shape.p0.y * yscale),
                p1=Vector2(shape.p1.x * xscale, shape.p1.y * yscale)
            )
        elif type(shape) is Drawings.Rect:
            return Drawings.Rect(
                p0=Vector2(shape.p0.x * xscale, shape.p0.y * yscale),
                p1=Vector2(shape.p1.x * xscale, shape.p1.y * yscale)
            )
        elif type(shape) is Rect:
            return Rect(
                x0=shape.x0 * xscale, y0=shape.y0 * yscale,
                x1=shape.x1 * xscale, y1=shape.y1 * yscale
            )
        elif type(shape) is Drawings.Curve:
            return Drawings.Curve(
                [
                    Vector2(point.x * xscale, point.y * yscale)
                    for point in shape.points
                ]
            )
        elif type(shape) is Drawings.Quad:
            return Drawings.Quad(
                p0=Vector2(shape.p0.x * xscale, shape.p0.y * yscale),
                p1=Vector2(shape.p1.x * xscale, shape.p1.y * yscale),
                p2=Vector2(shape.p2.x * xscale, shape.p2.y * yscale),
                p3=Vector2(shape.p3.x * xscale, shape.p3.y * yscale)
            )
        else:
            raise TypeError

    def draw_rect(
        self, img: Image.Image, rects: list[Drawings.Rect | BBox2D],
        color: ColorVar=(0,0,255), thickness: int=1
    ) -> Image.Image:
        if type(rects) is not list:
            rects = [rects]
        color = Convert.cast_color(color)
        img = Convert.pil_to_cv(img)
        for rect in rects:
            if type(rect) is Drawings.Rect:
                img = CvUtil.rectangle(
                    img, pt1=rect.p0, pt2=rect.p1,
                    color=color, thickness=thickness, lineType=cv2.LINE_AA
                )
            elif type(rect) is BBox2D:
                img = CvUtil.rectangle(
                    img, pt1=rect.v0, pt2=rect.v1,
                    color=color, thickness=thickness, lineType=cv2.LINE_AA
                )
            else:
                raise TypeError
        img = Convert.cv_to_pil(img)
        return img
    
    def draw_line(
        self, img: Image.Image, lines: list[Drawings.Line],
        color: ColorVar=(0,0,255), thickness: int=1
    ) -> Image.Image:
        if type(lines) is not list:
            lines = [lines]
        color = Convert.cast_color(color)
        img = Convert.pil_to_cv(img)
        for line in lines:
            # TODO: Move scaling out of this method.
            img = CvUtil.line(
                img, pt1=line.p0, pt2=line.p1,
                color=color, thickness=thickness, lineType=cv2.LINE_AA
            )
        img = Convert.cv_to_pil(img)
        return img

    def get_text(self, textType: str='text') -> list[str]:
        return self._page.get_text(textType).split('\n')

    def get_text_blocks(self, rotate: bool=True, scale: bool=True) -> tuple[list[BBox2D], list[str]]:
        data: list[float, float, float, float, str, int, int] = \
            self._page.get_text('blocks')
        # x0, y0, x1, y1, rawText, idx, _ = data[0]
        bboxList: list[BBox2D] = []
        rawTextList: list[str] = []
        xscale, yscale = self._dpiScaleFactor
        for x0, y0, x1, y1, rawText, idx, _ in data:
            rect = Rect(x0, y0, x1, y1)
            rawTextList.append(rawText)

            if rotate:
                rect = rect * self._page.rotation_matrix
            if scale:
                bbox = BBox2D(
                    Vector2(rect.x0 * xscale, rect.y0 * yscale),
                    Vector2(rect.x1 * xscale, rect.y1 * yscale)
                )
            else:
                bbox = BBox2D(
                    Vector2(rect.x0, rect.y0),
                    Vector2(rect.x1, rect.y1)
                )
            bboxList.append(bbox)

        return bboxList, rawTextList

    def get_drawings(self, rotate: bool=True, scale: bool=True) -> Drawings:
        """Note
        drawing['type']
            'f': fill-only path
            's': stroke-only path
            'fs': fill and stroke

        item types:
            'l': line
            'c': curve
            're': rectangle
            ?: quad
        """
        drawings = Drawings()
        _drawings: list[dict] = self._page.get_drawings()
        for drawing in _drawings:
            for obj in drawing['items']:
                item_type: str = obj[0]
                if item_type == 'l': # line
                    p0: Point = obj[1]
                    p1: Point = obj[2]
                    if rotate:
                        p0 *= self._page.rotation_matrix
                        p1 *= self._page.rotation_matrix
                    if scale:
                        xscale, yscale = self._dpiScaleFactor
                        drawings.lines.append(Drawings.Line(
                            Vector2(p0.x * xscale, p0.y * yscale),
                            Vector2(p1.x * xscale, p1.y * yscale)
                        ))
                    else:
                        drawings.lines.append(Drawings.Line(
                            Vector2(p0.x, p0.y),
                            Vector2(p1.x, p1.y)
                        ))
                elif item_type == 'c': # curve
                    points: list[Point] = list(obj)[1:]
                    if rotate:
                        points = [point * self._page.rotation_matrix for point in points]
                    if scale:
                        drawings.curves.append(Drawings.Curve([Vector2(p.x, p.y) for p in points]))
                    else:
                        xscale, yscale = self._dpiScaleFactor
                        drawings.curves.append(Drawings.Curve([Vector2(p.x * xscale, p.y * yscale) for p in points]))
                elif item_type == 're': # rectangle
                    rect: Rect = obj[1]
                    if rotate:
                        rect *= self._page.rotation_matrix
                    fill: int = obj[2] # Not sure what this is. Fill?
                    if scale:
                        xscale, yscale = self._dpiScaleFactor
                        drawings.rectangles.append(
                            Drawings.Rect(
                                p0=Vector2(rect.x0 * xscale, rect.y0 * yscale),
                                p1=Vector2(rect.x1 * xscale, rect.y1 * yscale)
                            )
                        )
                    else:
                        drawings.rectangles.append(
                            Drawings.Rect(
                                p0=Vector2(rect.x0, rect.y0),
                                p1=Vector2(rect.x1, rect.y1)
                            )
                        )
                elif item_type == 'qu': # Quad
                    quad: Quad = obj[1]
                    if rotate:
                        quad *= self._page.rotation_matrix
                    if scale:
                        xscale, yscale = self._dpiScaleFactor
                        drawings.quads.append(
                            Drawings.Quad(
                                p0=Vector2(quad.ul.x * xscale, quad.ul.y * yscale),
                                p1=Vector2(quad.ur.x * xscale, quad.ur.y * yscale),
                                p2=Vector2(quad.lr.x * xscale, quad.lr.y * yscale),
                                p3=Vector2(quad.ll.x * xscale, quad.ll.y * yscale)
                            )
                        )
                    else:
                        drawings.quads.append(
                            Drawings.Quad(
                                p0=Vector2(quad.ul.x, quad.ul.y),
                                p1=Vector2(quad.ur.x, quad.ur.y),
                                p2=Vector2(quad.lr.x, quad.lr.y),
                                p3=Vector2(quad.ll.x, quad.ll.y)
                            )
                        )
                else:
                    raise NotImplementedError(obj)
        return drawings

from ._parsed import Drawings, Resolution
from ._matching import LineMeta, LineMatch, LineMatches