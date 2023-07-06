from __future__ import annotations
from typing import TYPE_CHECKING
import io
import json
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if TYPE_CHECKING:
    from . import Results, BBoxResult
    from .._structs import Annotations, Annotation
    from .._dataset import Dataset

from ....util import SuppressStd, RedirectStdToVariable

def eval_with_pycocotools(
    self: Results, gt: Dataset
):
    with SuppressStd():
        cocoGt = COCO()
        cocoGt.dataset = gt.to_dict()
        cocoGt.createIndex()

        cocoDt = COCO()
        dtAnns = self.to_dict()
        annsImgIds = [ann['image_id'] for ann in dtAnns]
        assert set(annsImgIds) == (set(annsImgIds) & set(cocoGt.getImgIds())), \
            'Results do not correspond to current coco set'
        import copy
        cocoDt.dataset['categories'] = copy.deepcopy(cocoGt.dataset['categories'])
        for id, ann in enumerate(dtAnns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
            if not 'segmentation' in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2]*bb[3]
            ann['id'] = id+1
            ann['iscrowd'] = 0
        cocoDt.dataset['annotations'] = dtAnns
        cocoDt.createIndex()
        cocoeval = COCOeval( # Creates: self.params
            cocoGt=cocoGt,
            cocoDt=cocoDt,
            iouType='bbox'
        )
        cocoeval.evaluate() # Creates: self.evalImgs
        cocoeval.accumulate() # Creates: self.eval, Requires self.evalImgs
    
    summary = io.StringIO()
    with RedirectStdToVariable(stdout=summary):
        cocoeval.summarize()
    # summary = '\n'.join(summary.getvalue().split('\n')[:-1])
    summary = summary.getvalue()

    return cocoeval.evalImgs, summary

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in [
            np.int64,
            np.ndarray
        ]:
            return obj.tolist()
        else:
            return obj.__dict__

def eval_with_pycocotools_and_dump(
    self: Results, gt: Dataset,
    dumpDir: str
):
    os.makedirs(dumpDir, exist_ok=True)
    evalImgs, summary = self.eval_with_pycocotools(gt)
    json.dump(evalImgs, open(f'{dumpDir}/evalImgs.json', 'w'), cls=MyEncoder)
    open(f'{dumpDir}/summary.txt', 'w').write(summary)
    return evalImgs, summary