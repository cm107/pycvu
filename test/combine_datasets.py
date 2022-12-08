import os
import random
from pycvu.coco.object_detection import *

imgRoot0 = '.'
d0 = Dataset.load(f'datasetDump/hanko-nameDataset.json')

mainDir = '/home/clayton/workspace/prj/mediatech_poc2/main'
imgRoot1 = f'{mainDir}/train_hanko'
d1 = Dataset.load(f'{mainDir}/coco_train_hanko.json')

numImages = min(len(d0.images), len(d1.images))

for d in [d0, d1]:
    if len(d.images) > numImages:
        idxList = list(range(len(d.images)))
        keepIdx = random.sample(idxList, k=numImages)
        keepId = [d.images[idx].id for idx in keepIdx]
        d.filter(imgFilter=lambda img: img.id in keepId, reindex=True, applyToSelf=True)

print('Adjusting image paths...')
for images, imgRoot, useRelative in [
    (d0.images, imgRoot0, True),
    (d1.images, imgRoot1, True)
]:
    images: Images = images
    for image in images:
        path = os.path.abspath(f"{imgRoot}/{image.file_name}")
        if useRelative:
            path = os.path.relpath(path, imgRoot0)
        image.file_name = path

print('Combining...')
dataset = Dataset.combine([d0, d1], showPbar=True)

for ann in dataset.annotations.search(lambda ann: ann.category_id == 2):
    ann.category_id = 0
for ann in dataset.annotations.search(lambda ann: ann.category_id == 3):
    ann.category_id = 1

idxList = dataset.categories.index(dataset.categories.get(lambda cat: cat.id == 2))
del dataset.categories[idxList]
idxList = dataset.categories.index(dataset.categories.get(lambda cat: cat.id == 3))
del dataset.categories[idxList]

if dataset.categories.get(lambda cat: cat.name == 'hanko').id != 0:
    hankoCat = dataset.categories.get(lambda cat: cat.name == 'hanko')
    nameCat = dataset.categories.get(lambda cat: cat.name == 'name')
    hankoAnns = dataset.annotations.search(lambda ann: ann.category_id == hankoCat.id)
    nameAnns = dataset.annotations.search(lambda ann: ann.category_id == nameCat.id)
    for ann in hankoAnns:
        ann.category_id = 0
    hankoCat.id = 0
    for ann in nameAnns:
        ann.category_id = 1
    nameCat.id = 1

for image in dataset.images:
    image.coco_url = image.file_name

dataset.save(f'datasetDump/mixed_hanko-nameDataset.json')