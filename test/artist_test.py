import os
import numpy as np
from pycvu import Artist, Interval, Color, HSV, Vector
from pycvu.util import LoadableImageMaskHandler, \
    CvUtil, PilUtil
from pycvu.text_generator import TextGenerator, TextSampler, \
    CharacterSets, StringSets

imgHandlerPath = "imgHandler.json"
if not os.path.isfile(imgHandlerPath):
    imgHandler = LoadableImageMaskHandler.from_wildcard(
        "symbol/*.png",
        Interval[HSV](HSV(0,0,0), HSV(359.9, 1, 0.9))
    )
    imgHandler.load_data()
    imgHandler.save(imgHandlerPath, includeData=True)
else:
    imgHandler = LoadableImageMaskHandler.load(imgHandlerPath)
imgHandlerRef = Artist.context.register_variable(imgHandler)

artist = Artist(np.ones((800, 800, 3), dtype=np.uint8) * 255)

"""
Note: In Kume's config, iters
is the number of repetitions while n_connect is the
length of each text string?

wtf is n_merge???
"""

# Proc 0
Artist.fontScale = Interval[float](0.4, 0.5)
Artist.thickness = Interval[int](1, 2)
# Artist.color = Interval[Color](Color(0, 0, 0), Color(50, 50, 50)) # hsv would be better
# Kume was setting all color channels to the same value. Doing this with hsv should have the same effect.
Artist.color = Interval[HSV](HSV(0,0,0), HSV(0,0,0.2))
artist.text(
    text=TextGenerator(
        characters=CharacterSets.alpha,
        textLength=1, # n_connect
        allowRepetition=True
    ),
    org=CvUtil.Callback.get_position_interval,
    repeat=50 # iters
)

# Proc 1
Artist.PIL.fontSize = Interval[int](10, 15)
artist.pil.text(
    text=TextGenerator(
        characters=CharacterSets.kana,
        textLength=1, # n_connect
        allowRepetition=True
    ),
    position=PilUtil.Callback.get_position_interval,
    rotation=Interval[float](-20, 20),
    repeat=50 # iters
)

# Proc 2
artist.pil.text(
    text=TextGenerator(
        characters=CharacterSets.kanji,
        textLength=1, # n_connect
        allowRepetition=True
    ),
    position=PilUtil.Callback.get_position_interval,
    rotation=Interval[float](-20, 20),
    repeat=50 # iters
)

# Proc 2
artist.pil.text(
    text=TextGenerator(
        characters=CharacterSets.kanji,
        textLength=Interval[int](2, 10), # Kume had this fixed at 10
        allowRepetition=True
    ),
    position=PilUtil.Callback.get_position_interval,
    rotation=Interval[float](-5, 5),
    repeat=10 # iters
)

# Proc 3
artist.pil.text(
    text=TextSampler(StringSets.namae),
    position=PilUtil.Callback.get_position_interval,
    direction='rtl',
    rotation=Interval[float](-20, 20),
    repeat=50 # iters
)

Artist.maskSetting.track = True

# Proc 4
artist.pil.text(
    text=TextSampler(StringSets.namae),
    position=PilUtil.Callback.get_position_interval,
    direction='ttb',
    rotation=Interval[float](-20, 20),
    repeat=20 # iters
)

# Proc 5
Artist.PIL.fontSize = Interval[int](15, 30)
Artist.PIL.hankoIsVertical = True
Artist.PIL.hankoMarginRatio = Interval[float](0.0, 0.2)
Artist.PIL.hankoOutlineWidthRatio = Interval[float](0.05, 0.3)
artist.pil.hanko(
    text=TextSampler(StringSets.namae),
    position=PilUtil.Callback.get_position_interval,
    rotation=Interval[float](-20, 20),
    repeat=20 # iters
)

Artist.maskSetting.track = False

# Proc 6 & 7
artist.line(
    pt1=CvUtil.Callback.get_position_interval,
    pt2=CvUtil.Callback.get_position_interval,
    repeat=1+3
)

# Proc 8
artist.circle(
    center=CvUtil.Callback.get_position_interval,
    radius=Interval[int](10, 50),
    repeat=30
)

# Proc 9

artist.ellipse(
    center=CvUtil.Callback.get_position_interval,
    axis=Interval[Vector[int]](Vector[int](100, 100), Vector[int](200, 200)), # Kume called this 'scale'. Incoherent naming.
    angle=Interval[float](-20, 20), # rotation
    repeat=0 # Removed from config??
)

# Proc 10
Artist.thickness = Interval[int](1, 3)
# TODO: rotation: [-10, 10]
artist.rectangle(
    pt1=CvUtil.Callback.get_position_interval,
    pt2=CvUtil.Callback.get_position_interval,
    repeat=10
)

Artist.maskSetting.track = True
# Proc 11
# TODO: scale: [0.9, 1.1]
# TODO: rotation: [-5, 5]
# TODO: noise: [-10, 20]
artist.overlay_image(
    foreground=imgHandler[0],
    position=CvUtil.Callback.get_position_interval,
    repeat=1
)

# Proc 12
# TODO: scale: [0.9, 1.1]
# TODO: rotation: [-5, 5]
# TODO: noise: [-10, 20]
artist.overlay_image(
    foreground=imgHandler[1],
    position=CvUtil.Callback.get_position_interval,
    repeat=1
)

# Proc 12
# TODO: scale: [0.9, 1.1]
# TODO: rotation: [-5, 5]
# TODO: noise: [-10, 20]
artist.overlay_image(
    foreground=imgHandler[3:7],
    position=CvUtil.Callback.get_position_interval,
    repeat=1
)

# Proc 13
# TODO: scale: [0.9, 1.1]
# TODO: rotation: [-5, 5]
# TODO: noise: [-20, 0]
artist.overlay_image(
    foreground=imgHandler[10],
    position=CvUtil.Callback.get_position_interval,
    repeat=2
)

# Proc 14
# TODO: scale: [0.9, 1.1]
# TODO: rotation: [-5, 5]
# TODO: noise: [30, 50]
artist.overlay_image(
    foreground=imgHandler[10],
    position=CvUtil.Callback.get_position_interval,
    repeat=1
)

Artist.maskSetting.track = False

# Proc 15
# type: "dest"
# What is this?

artist.save('/tmp/artistTestSave.json', saveImg=False, saveMeta=True)

result, maskHandler = artist.draw_and_get_masks()

from shutil import rmtree
import cv2

previewDump = 'artistTestDump'
if os.path.isdir(previewDump):
    rmtree(previewDump)
os.makedirs(previewDump, exist_ok=True)
cv2.imwrite(f"{previewDump}/result.png", result)
cv2.imwrite(f"{previewDump}/maskPreview.png", maskHandler.preview)
for i, mask in enumerate(maskHandler):
    if mask._mask.sum() == 0:
        continue
    maskImg = mask.get_preview(showBBox=True, showContours=True, minNumPoints=6)
    numStr = str(i)
    while len(numStr) < 2:
        numStr = f"0{numStr}"
    cv2.imwrite(f"{previewDump}/mask{numStr}.png", maskImg)

# Color is weird. Need to figure out what kume is doing with this (10, 50) color range.