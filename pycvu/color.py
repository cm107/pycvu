from __future__ import annotations
import cv2
import numpy as np
import numpy.typing as npt
from .base import Base

class Color(Base):
    def __init__(self, r: float=0, g: float=0, b: float=0, a: float=None, _scale: float=255):
        self.r = r; self.g = g; self.b = b
        if a is None:
            self.a = _scale
        else:
            self.a = a
        self._scale = _scale

    def __str__(self) -> str:
        innerStr = ', '.join([
            f'{key}={int(val)}'
            for key, val in self.__dict__.items()
            if (
                not key.startswith('_')
                and not (
                    key == 'a'
                    and self.a == self._scale
                )
            )
        ])
        return f"{type(self).__name__}({innerStr})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def scale(self) -> float:
        return self._scale
    
    @scale.setter
    def scale(self, value: float):
        ratio = value / self._scale
        self.r *= ratio
        self.g *= ratio
        self.b *= ratio
        self.a *= ratio
        self._scale = value

    def rescale(self, newScale: float) -> Color:
        color = self.deepcopy()
        color.scale = newScale
        return color

    @property
    def rgb(self) -> tuple[float, float, float]:
        return (self.r, self.g, self.b)
    
    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    @property
    def bgr(self) -> tuple[float, float, float]:
        return (self.b, self.g, self.r)
    
    @property
    def bgra(self) -> tuple[float, float, float, float]:
        return (self.b, self.g, self.r, self.a)
    
    def to_hsv(self) -> HSV:
        return HSV.from_color(self)
    
    @classmethod
    def from_hsv(cls, hsv: HSV) -> Color:
        return hsv.to_color()
    
    @property
    def hsv(self) -> tuple[float, float, float]:
        hsv = self.to_hsv()
        return (hsv.h, hsv.s, hsv.v)

    @classmethod
    @property
    def red(cls) -> Color:
        return cls(255, 0, 0)
    
    @classmethod
    @property
    def green(cls) -> Color:
        return cls(0, 255, 0)
    
    @classmethod
    @property
    def blue(cls) -> Color:
        return cls(0, 0, 255)

    @classmethod
    @property
    def black(cls) -> Color:
        return cls(0, 0, 0)
    
    @classmethod
    @property
    def white(cls) -> Color:
        return cls(255, 255, 255)

    @classmethod
    @property
    def yellow(cls) -> Color:
        return cls(255, 255, 0)

    @classmethod
    @property
    def magenta(cls) -> Color:
        return cls(255, 0, 255)
    
    @classmethod
    @property
    def cyan(cls) -> Color:
        return cls(0, 255, 255)

    @classmethod
    @property
    def orange(cls) -> Color:
        return cls(255, 165, 0)

    def sample_img(self, shape: tuple) -> npt.NDArray[np.uint8]:
        img = np.zeros(shape, dtype=np.uint8)
        img[:,:] = self.bgr
        return img

    @property
    def preview_img(self) -> npt.NDArray[np.uint8]:
        return self.sample_img((500, 500, 3))

    @property
    def distant_color_sample(self) -> Color:
        hsv = self.to_hsv()
        if hsv.s == 0:
            hsv.s = 1
            hsv.v = 1 - hsv.v
        return hsv.rotate(180).to_color()

    def preview(self):
        cv2.imshow('preview', self.preview_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    @staticmethod
    def debug():
        color0 = Color(10, 123, 174)
        color1 = color0.distant_color_sample
        print(f"{color0=}, {color1=}")
        preview = np.concatenate(
            [color0.preview_img, color1.preview_img],
            axis=1, dtype=np.uint8
        )
        cv2.imshow('preview', preview)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

class HSV(Base):
    def __init__(self, h: float, s: float, v: float):
        self.h = h
        """Hue. Range: 0 ~ 360 (Corresponds to an angle.)"""

        self.s = s
        """Saturation. Range: 0 ~ 1"""
        
        self.v = v
        """Value. Range: 0 ~ 1"""
    
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.h},{self.s},{self.v})"

    def rotate(self, angle: float) -> HSV:
        return HSV(
            h=(self.h + angle) % 360,
            s=self.s, v=self.v
        )

    def to_color(self) -> Color:
        """
        Refer to https://www.rapidtables.com/convert/color/hsv-to-rgb.html
        """
        c = self.v * self.s
        x = c * (1 - abs((self.h / 60) % 2 - 1))
        m = self.v - c
        assert 0 <= self.h < 360
        if self.h < 60:
            return Color(c+m, x+m, 0+m, _scale=1).rescale(255)
        elif self.h < 120:
            return Color(x+m, c+m, 0+m, _scale=1).rescale(255)
        elif self.h < 180:
            return Color(0+m, c+m, x+m, _scale=1).rescale(255)
        elif self.h < 240:
            return Color(0+m, x+m, c+m, _scale=1).rescale(255)
        elif self.h < 300:
            return Color(x+m, 0+m, c+m, _scale=1).rescale(255)
        elif self.h < 360:
            return Color(c+m, 0+m, x+m, _scale=1).rescale(255)
        else:
            raise Exception
    
    @classmethod
    def from_color(cls, color: Color) -> HSV:
        """
        Refer to https://www.rapidtables.com/convert/color/rgb-to-hsv.html
        """
        r, g, b = color.rescale(1).rgb
        cmin = min(r, g, b); cmax = max(r, g, b)
        delta = cmax - cmin
        if delta == 0:
            h = 0
        elif r == cmax:
            h = 60 * (((g - b) / delta) % 6)
        elif g == cmax:
            h = 60 * (((b - r) / delta) + 2)
        elif b == cmax:
            h = 60 * (((r - g) / delta) + 4)
        else:
            raise Exception
        if cmax == 0:
            s = 0
        else:
            s = delta / cmax
        v = cmax
        return HSV(h, s, v)
    
    @staticmethod
    def unit_test():
        # hsv to rgb
        hsvToRgbMap: dict = {
            (0, 0, 0): (0, 0, 0), # Black
            (0, 0, 1): (255, 255, 255), # white
            (0, 1, 1): (255, 0, 0), # red
            (120, 1, 1): (0, 255, 0), # lime
            (240, 1, 1): (0, 0, 255), # blue
            (60, 1, 1): (255, 255, 0), # yellow
            (180, 1, 1): (0, 255, 255), # cyan
            (300, 1, 1): (255, 0, 255), # magenta
            (0, 0, 0.75): (191, 191, 191), # silver
            (0, 0, 0.5): (128, 128, 128), # gray
            (0, 1, 0.5): (128, 0, 0), # maroon
            (60, 1, 0.5): (128, 128, 0), # olive
            (120, 1, 0.5): (0, 128, 0), # green
            (300, 1, 0.5): (128, 0, 128), # purple
            (180, 1, 0.5): (0, 128, 128), # teal
            (240, 1, 0.5): (0, 0, 128) # navy
        }
        for hsv, rgb in hsvToRgbMap.items():
            calcRGB = HSV(*hsv).to_color().rgb
            calcHSV = Color(*rgb).hsv
            calcRGB = tuple([round(val) for val in calcRGB])
            calcHSV = list(calcHSV)
            calcHSV[0] = round(calcHSV[0])
            calcHSV[1] = round(calcHSV[1], 2)
            calcHSV[2] = round(calcHSV[2], 2)
            calcHSV = tuple(calcHSV)
            assert calcRGB == rgb, f"Failed HSV{hsv} -> RGB{calcRGB} != RGB{rgb}"
            assert calcHSV == hsv, f"Failed RGB{rgb} -> HSV{calcHSV} != HSV{hsv}"

        print("HSV Unit Test Passed")
