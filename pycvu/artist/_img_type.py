import numpy as np
from PIL import Image as pilImage
from ..base import BaseEnum

class ImageType(BaseEnum):
    CV = np.ndarray
    PIL = pilImage.Image
