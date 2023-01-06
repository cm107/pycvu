import os
import fitz
from fitz.fitz import Document, Page
# from multiprocessing import Pool
import multiprocessing as mp
from cairosvg import svg2png
import glob
from tqdm import tqdm
from functools import partial
import traceback

__all__ = [
    'pdf_to_png'
]

def _pdf_to_png(
    path: str,
    output_dir: str='pdf_images', dpi: int=256,
    skip_existing: bool=True, background_color: str=None
):
    ext: str = 'png'
    doc: Document = fitz.open(path)
    nDigits = len(str(len(doc) - 1))
    os.makedirs(output_dir, exist_ok=True)
    if skip_existing:
        existingSaved: list[str] = glob.glob(f"{os.path.basename(path)}.*.{ext}")
        if len(existingSaved) == len(doc):
            return
    for page, content in enumerate(doc):
        content: Page = content
        bytestring: str = content.get_svg_image()
        pageStr = str(page)
        if len(pageStr) < nDigits:
            pageStr = f"0{pageStr}"
        filename = f"{os.path.basename(path)}.{pageStr}.{ext}"
        savePath = f"{output_dir}/{filename}"
        try:
            svg2png(
                bytestring=bytestring,
                write_to=savePath,
                dpi=dpi,
                background_color=background_color
            )
        except Exception as e:
            with open(f"{output_dir}/{os.path.basename(path)}.{pageStr}.txt", 'w') as f:
                tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
                f.write(tb_str)

def pdf_to_png(
    path: str,
    output_dir: str='pdf_images', dpi: int=256,
    skip_existing: bool=True, background_color: str=None,
    pool: int=None, save_cpus: int=2, showPbar: bool=True
):
    _func: partial = partial(
        _pdf_to_png,
        output_dir=output_dir,
        dpi=dpi,
        skip_existing=skip_existing,
        background_color=background_color
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
