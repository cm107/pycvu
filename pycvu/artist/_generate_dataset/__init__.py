from __future__ import annotations
from typing import TYPE_CHECKING, Type, Callable
if TYPE_CHECKING:
    from .._artist import Artist
import os
import glob
from shutil import rmtree
import numpy as np
from ...coco.object_detection import Dataset as ObjectDetectionDataset
from ...coco.ocr import Dataset as OcrDataset
from ...mask import MaskHandler

from ._object_detection import _generate_dataset_object_detection
from ._ocr import _generate_dataset_ocr

def generate_dataset(
    self: Artist, frames: int, dumpDir: str="artist_dataset_dump",
    showPbar: bool=True, repeat: int=1,
    combineResults: bool=True,
    useOcrFormat: bool=False,
    resultCallback: Callable[[np.ndarray, MaskHandler], None]=None,
    batchSize: int=1, saveCpus: int=2
):
    assert len(self._drawQueue) > 0, f"Nothing has been queued for drawing yet."
    if repeat > 1:
        if not os.path.isdir(dumpDir):
            os.makedirs(dumpDir)
            currentIter = 0
        else:
            datasetDirPaths = [
                path
                for path in glob.glob(f"{dumpDir}/dataset*")
                if os.path.isdir(path)
            ]
            iterNums = [int(os.path.basename(path).replace('dataset', '')) for path in datasetDirPaths]
            lastIter = max(iterNums)
            lastDatasetDir = datasetDirPaths[iterNums.index(lastIter)]
            if not os.path.isfile(f"{lastDatasetDir}/dataset.json"):
                # Unfinished. Needs to be redone.
                rmtree(lastDatasetDir)
                currentIter = lastIter
            else:
                # Resume from next iteration.
                currentIter = lastIter + 1
        
        collectionWasUpdated = False
        for k in range(currentIter, repeat):
            kStr = str(k)
            while len(kStr) < 3:
                kStr = f"0{kStr}"
            if not useOcrFormat:
                _generate_dataset_object_detection(
                    self,
                    frames=frames, dumpDir=f"{dumpDir}/dataset{kStr}",
                    showPbar=showPbar,
                    resultCallback=resultCallback,
                    batchSize=batchSize, saveCpus=saveCpus
                )
            else:
                _generate_dataset_ocr(
                    self,
                    frames=frames, dumpDir=f"{dumpDir}/dataset{kStr}",
                    showPbar=showPbar,
                    resultCallback=resultCallback,
                    batchSize=batchSize, saveCpus=saveCpus
                )
            collectionWasUpdated = True
        
        if combineResults:
            combinedDatasetPath = f"{dumpDir}/dataset.json"
            if not os.path.isfile(combinedDatasetPath) or collectionWasUpdated:
                datasetPaths = sorted(glob.glob(f"{dumpDir}/dataset*/dataset.json"))
                assert len(datasetPaths) > 1
                if not useOcrFormat:
                    combined = ObjectDetectionDataset.combine(datasetPaths, showPbar=showPbar)
                else:
                    raise NotImplementedError
                combined.save(combinedDatasetPath)

    else:
        if not useOcrFormat:
            _generate_dataset_object_detection(
                self,
                frames=frames, dumpDir=dumpDir,
                showPbar=showPbar,
                resultCallback=resultCallback,
                batchSize=batchSize, saveCpus=saveCpus
            )
        else:
            _generate_dataset_ocr(
                self,
                frames=frames, dumpDir=dumpDir,
                showPbar=showPbar,
                resultCallback=resultCallback,
                batchSize=batchSize, saveCpus=saveCpus
            )