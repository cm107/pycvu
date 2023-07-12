from __future__ import annotations
import os
import glob
import json
import fitz
import numpy as np
from PIL import Image

from pyevu import BBox2D, Vector2, Interval
from ..util import Convert

class RawChar:
    def __init__(
        self,
        origin: tuple[float, float],
        bbox: BBox2D,
        c: str
    ):
        self.origin = origin
        self.bbox = bbox
        self.c = c

    @classmethod
    def from_raw(cls, raw) -> RawChar:
        return RawChar(
            origin=raw['origin'],
            bbox=BBox2D(
                Vector2.FromList(raw['bbox'][:2]),
                Vector2.FromList(raw['bbox'][2:])
            ),
            c=raw['c'],
        )

class RawDataCharsSpan:
    def __init__(
        self,
        size: float, # font size?
        flags: int,
        font: str,
        color: int,
        ascender: float,
        descender: float,
        chars: list[RawChar]
    ):
        self.size = size
        self.flags = flags
        self.font = font
        self.color = color
        self.ascender = ascender
        self.descender = descender
        self.chars = chars

    @classmethod
    def from_raw(cls, raw) -> RawDataCharsSpan:
        return RawDataCharsSpan(
            size=raw['size'],
            flags=raw['flags'],
            font=raw['font'],
            color=raw['color'],
            ascender=raw['ascender'],
            descender=raw['descender'],
            chars=[RawChar.from_raw(val) for val in raw['chars']]
        )

class RawDataBlockLine:
    def __init__(
        self,
        spans: list[RawDataCharsSpan],
        wmode: int, dir: tuple[float, float],
        bbox: BBox2D
    ):
        self.spans = spans
        self.wmode = wmode
        self.dir = dir
        self.bbox = bbox

    @classmethod
    def from_raw(cls, raw) -> RawDataBlockLine:
        return RawDataBlockLine(
            spans=[
                RawDataCharsSpan.from_raw(val)
                for val in raw['spans']
                if 'chars' in val
            ],
            wmode=raw['wmode'],
            dir=raw['dir'],
            bbox=BBox2D(
                Vector2.FromList(raw['bbox'][:2]),
                Vector2.FromList(raw['bbox'][2:])
            )
        )

class RawDataBlock:
    def __init__(
        self,
        number: int, type: int,
        bbox: BBox2D,
        lines: list[RawDataBlockLine]
    ):
        self.number = number
        self.type = type
        self.bbox = bbox
        self.lines = lines

    @classmethod
    def from_raw(cls, raw) -> RawDataBlock:
        return RawDataBlock(
            number=raw['number'], type=raw['type'],
            bbox=BBox2D(
                Vector2.FromList(raw['bbox'][:2]),
                Vector2.FromList(raw['bbox'][2:])
            ),
            lines=[RawDataBlockLine.from_raw(val) for val in raw['lines']]
        )

class RawData:
    def __init__(
        self, width: float, height: float,
        blocks: list[RawDataBlock]
    ):
        self.width = width
        self.height = height
        self.blocks = blocks
    
    @classmethod
    def from_raw(cls, raw) -> RawData:
        return RawData(
            width=raw['width'], height=raw['height'],
            blocks=[RawDataBlock.from_raw(val) for val in raw['blocks']]
        )

class Block:
    def __init__(
        self,
        x0: float, y0: float, x1: float, y1: float,
        text: str,
        block_no: int, block_type: int
    ):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.text = text
        self.block_no = block_no
        self.block_type = block_type

    def __str__(self) -> str:
        coordStr = ','.join([str(int(val)) for val in [self.x0, self.y0, self.x1, self.y1]])
        coordStr = f"({coordStr})"
        return f"{coordStr}: {repr(self.text)}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, item_dict: dict) -> Block:
        return Block(**item_dict)
    
    def save(self, path: str):
        json.dump(self.to_dict(), open(path, 'w'), ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> Block:
        return cls.from_dict(json.load(open(path, 'r')))

    @staticmethod
    def save_batch(blocks: list[Block], path: str):
        json.dump(
            [block.to_dict() for block in blocks],
            open(path, 'w'), ensure_ascii=False
        )

    @staticmethod
    def load_batch(path: str) -> list[Block]:
        return [
            Block.from_dict(item_dict)
            for item_dict in json.load(open(path, 'r'))
        ]

    @property
    def rect(self) -> fitz.Rect:
        return fitz.Rect(self.x0, self.y0, self.x1, self.y1)
    
    @property
    def bbox(self) -> BBox2D:
        return BBox2D(Vector2(self.x0, self.y0), Vector2(self.x1, self.y1))

    @bbox.setter
    def bbox(self, value: BBox2D):
        self.x0 = value.v0.x; self.x1 = value.v1.x
        self.y0 = value.v0.y; self.y1 = value.v1.y

    @staticmethod
    def same_row(block0: Block, block1: Block, sameRowThresh: float=0.5) -> bool:
        bbox0 = block0.bbox; bbox1 = block1.bbox
        y0 = bbox0.yInterval; y1 = bbox1.yInterval
        yint = Interval.Intersection(y0, y1)
        return (
            yint is not None
            and yint.length / y0.length >= sameRowThresh
            and yint.length / y1.length >= sameRowThresh
        )

def get_row_idx(block: Block, rows: list[list[Block]]) -> int:
    for i, row in enumerate(rows):
        if Block.same_row(block, row[0]):
            return i
    return len(rows)

def get_rows(blocks: list[Block], applySort: bool=True) -> list[list[Block]]:
    rows: list[list[Block]] = []
    for block in blocks:
        idx = get_row_idx(block, rows)
        if idx < len(rows):
            rows[idx].append(block)
        else:
            rows.append([block])
    if applySort:
        for row in rows:
            row.sort(key=lambda block: block.x0)
        rows.sort(key=lambda row: row[0].y0)
    return rows      

def split_image_by_height(
    img: np.ndarray | Image.Image,
    maxAspect: float=0.5,
    textRows: list[list[Block]]=None
) -> list[tuple[Image.Image, int, int]]:
    """
    Return:
        list of image, yOffset, relevantHeight tuples
    """
    if type(img) is Image.Image:
        img = Convert.pil_to_cv(img)
    h, w = img.shape[:2]
    originalAr = h / w
    if originalAr > maxAspect:
        relevantHeight = int(w * maxAspect)
        
        result: list[tuple[np.ndarray, int]] = []
        ymin: int = None; ymax: int = None
        while ymax is None or ymax < h:
            ymin = 0 if ymin is None else ymax
            ymax = relevantHeight if ymax is None else ymax + relevantHeight
            if ymax > h:
                ymax = h       

            if textRows is not None:
                for row in textRows:
                    rowY = row[0].bbox.yInterval
                    if rowY.Contains(ymax) and ymax < rowY.min:
                        ymax = rowY.min
                        break

            assert ymax - ymin > 0, f"{ymin=}, {ymax=}, {ymax-ymin=}"
            croppedImg = img[ymin:ymax, :, :]
            result.append((Convert.cv_to_pil(croppedImg.copy()), ymin, ymax - ymin))
        return result
    else:
        result = [(Convert.cv_to_pil(img), 0, h)]
    
    return result

def get_pdf_paths(pdfDir: str, recursive: bool=False, workingPaths: list[str]=None):
    if not recursive:
        return glob.glob(f"{pdfDir}/*.pdf")
    else:
        workingPaths = workingPaths if workingPaths is not None else []
        for _path in glob.glob(f"{pdfDir}/*"):
            if os.path.isfile(_path) and _path.endswith('.pdf'):
                workingPaths.append(_path)
            elif os.path.isdir(_path):
                get_pdf_paths(pdfDir=_path, recursive=True, workingPaths=workingPaths)
            else:
                pass
        return workingPaths

class Parser:
    def __init__(self, pdfPath: str):
        self.doc: fitz.Document = fitz.open(pdfPath)

    def get_image(self, pageNum: int, scale: float=1.0) -> Image.Image:
        page: fitz.Page = self.doc[pageNum]
        pix = page.get_pixmap(
            matrix=fitz.Matrix(*(scale,scale))
        )
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def get_combined_image(self, scale: float=1.0) -> Image.Image:
        frames: list[Image.Image] = []
        for pageNum in range(len(self.doc)):
            img = self.get_image(pageNum=pageNum, scale=scale)
            frames.append(img)
        return Convert.cv_to_pil(
            np.concatenate(
                [Convert.pil_to_cv(frame) for frame in frames],
                axis=0
            )
        )

    def parse_rows(self, pageNum: int, scale: float=1.0) -> list[list[Block]]:
        page: fitz.Page = self.doc[pageNum]
        textpage = page.get_textpage()
        rawData = RawData.from_raw(textpage.extractRAWDICT())
        blocks = [
            Block(
                x0=char.bbox.v0.x * scale, y0=char.bbox.v0.y * scale,
                x1=char.bbox.v1.x * scale, y1=char.bbox.v1.y * scale,
                text=char.c,
                block_no=0, block_type=0
            )
            for b in rawData.blocks
            for line in b.lines
            for span in line.spans
            for char in span.chars
        ]
        rows = get_rows(blocks)
        return rows

    def parse_image_blocks(self, pageNum: int, scale: float=1.0) -> list[Block]:
        page: fitz.Page = self.doc[pageNum]
        pageImages = page.get_images()
        pageImageNumbers = [pageImage[0] for pageImage in pageImages]
        pageImageNames = [pageImage[7] for pageImage in pageImages]
        imageBoxes = [page.get_image_bbox(pageImageName) for pageImageName in pageImageNames]
        return [
            Block(
                x0=box.x0 * scale, y0=box.y0 * scale, x1=box.x1 * scale, y1=box.y1 * scale,
                text=name, block_no=num, block_type=1
            )
            for num, name, box in zip(pageImageNumbers, pageImageNames, imageBoxes)
        ]
