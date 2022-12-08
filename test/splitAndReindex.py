import os
from pycvu.coco.object_detection import *
dumpDir = 'datasetDump'
dataset = Dataset.load(f'{dumpDir}/dataset.json')

from pyevu import BBox2D, Vector2

def isValidAnn(ann: Annotation) -> bool:
    bbox = BBox2D(Vector2(*ann.bbox[:2]), Vector2(*ann.bbox[:2]) + Vector2(*ann.bbox[2:]))
    if bbox.area <= 0:
        return False
    if bbox.v0.x >= bbox.v1.x or bbox.v0.y >= bbox.v1.y:
        return False
    return True

# for name in ['waku', 'hanko', 'name']:
for names in [
    ['waku'],
    ['hanko'],
    ['name'],
    ['hanko', 'name']
]:
    namesStr = '-'.join(names)
    print(namesStr)
    savePath = f"{dumpDir}/{namesStr}Dataset.json"
    if os.path.isfile(savePath):
        continue
    d = dataset.filter(
        catFilter=lambda cat: cat.name in names,
        annFilter=lambda ann: isValidAnn(ann),
        reindex=True, showPbar=True, leavePbar=True
    )
    for ann in d.annotations:
        ann.segmentation = []
    d.save(savePath)
