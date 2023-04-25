from __future__ import annotations
import os
import cv2
import glob
from tqdm import tqdm
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Dataset

def generate_easyocr_recognition_labels(
    self: Dataset, dumpDir: str,
    imgDir: str=None,
    showPbar: bool=False, leavePbar: bool=False
):
    data = {
        'filename': [],
        'words': []
    }
    images = self.images
    if showPbar:
        images = tqdm(images, desc="Generating Recognition Labels", leave=leavePbar)
    
    # Delete existing dump files, if any
    if os.path.isdir(dumpDir):
        for _path in glob.glob(f"{dumpDir}/*.png") + glob.glob(f"{dumpDir}/*.csv"):
            os.remove(_path)
    
    # Create dump directory if it doesn't exist yet
    os.makedirs(dumpDir, exist_ok=True)

    # Crop and save text images
    for image in images:
        imgFilename = os.path.basename(image.file_name)
        if imgDir is None:
            imgPath = image.file_name
        else:
            imgPath = f"{imgDir}/{imgFilename}"
        basename = os.path.splitext(imgFilename)[0]

        img = cv2.imread(imgPath)
        anns = self.annotations.search(lambda ann: ann.image_id == image.id)
        for i, ann in enumerate(anns):
            bbox = ann.bbox2d
            croppedImg = bbox.crop_image(img)
            text = ann.text
            saveFilename = f"{basename}-{i}.png"
            savePath = f"{dumpDir}/{saveFilename}"
            cv2.imwrite(savePath, croppedImg)
            data['filename'].append(saveFilename)
            data['words'].append(text)
    
    df = pd.DataFrame(data)
    df.to_csv(f"{dumpDir}/labels.csv", header=True, index=False)

def generate_localization_transcription_gt(
    self: Dataset, dumpDir: str,
    showPbar: bool=False, leavePbar: bool=False
):
    images = self.images
    if showPbar:
        images = tqdm(images, desc="Generating Localization Transcription GT", leave=leavePbar)
    
    # Delete existing dump files, if any
    if os.path.isdir(dumpDir):
        for _path in glob.glob(f"{dumpDir}/*.txt"):
            os.remove(_path)
    
    # Create dump directory if it doesn't exist yet
    os.makedirs(dumpDir, exist_ok=True)

    # Crop and save text images
    for image in images:
        imgFilename = os.path.basename(image.file_name)
        basename = os.path.splitext(imgFilename)[0]
        savePath = f"{dumpDir}/{basename}.txt"
        anns = self.annotations.search(lambda ann: ann.image_id == image.id)
        if len(anns) == 0:
            continue
        with open(savePath, 'w') as f:
            for i, ann in enumerate(anns):
                lineStr = ','.join([str(int(val)) for val in ann.quad])
                lineStr += f",{ann.text}"
                if i == 0:
                    f.write(lineStr)
                else:
                    f.write('\n' + lineStr)
        f.close()