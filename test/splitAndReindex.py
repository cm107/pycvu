from pycvu.coco.object_detection import *
dumpDir = 'datasetDump'
dataset = Dataset.load(f'{dumpDir}/dataset.json')

for name in ['waku', 'hanko', 'name']:
    d = dataset.filter(catFilter=lambda cat: cat.name == name, reindex=False, showPbar=True, leavePbar=True)
    d.save(f"{dumpDir}/{name}Dataset.json")
