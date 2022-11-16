from joblib import Parallel, delayed
from functools import partial
from pycvu import Interval
from pycvu.util import IntVar, Convert, CvUtil
from pycvu.color import Color
from pycvu.text_generator import TextGenerator, CharacterSets
import numpy as np
import cv2
import time

test = 2

if test == 0:
    def pow(value: float, power: float) -> float:
        return value**power

    def square(value: float) -> float:
        return pow(value, 2)

    cube: partial = partial(pow, power=3)

    print(Parallel(n_jobs=2)(delayed(pow)(i, 2) for i in range(10)))
    print(Parallel(n_jobs=2)(delayed(square)(i) for i in range(10)))
    print(Parallel(n_jobs=2)(delayed(cube)(i) for i in range(10)))
elif test == 1:
    class Settings:
        def __init__(self, do_add: bool=False, do_multiply: bool=False, repeat: IntVar=1):
            self.do_add = do_add
            self.do_multiply = do_multiply
            self.repeat = repeat
        
        def apply(self, value: float) -> float:
            repeat = Convert.cast_builtin(self.repeat)
            for i in range(repeat):
                if self.do_add:
                    value += 5
                if self.do_multiply:
                    value *= 2.5
            return value
        
    class Worker:
        def __init__(self):
            self.values = [1,2,3,4,5,6,7,8,9,10]
            self.settings = Settings(do_add=True, do_multiply=True, repeat=Interval[int](3, 5))

        def apply_settings(self, value: float, offset: float=0):
            value = self.settings.apply(value)
            value += offset
            return value
        
        def apply_all(self):
            f = partial(self.apply_settings, offset=0.1)
            print(Parallel(n_jobs=2)(delayed(f)(i) for i in self.values))
    
    worker = Worker()
    worker.apply_all()
elif test == 2:
    class Worker:
        def __init__(self):
            self.img = np.zeros((500, 500, 3))
            self._draw_queue: list[partial] = []
            self.radius = Interval[int](50, 100)
            self.color = Interval[Color](Color(0,0,0), Color(100, 100, 100))
            self.thickness = Interval[int](1, 2)
            self.lineType = cv2.LINE_AA
            self.text_gen = TextGenerator(
                characters=CharacterSets.alpha,
                textLength=Interval[int](1,3),
                allowRepetition=True
            )

        def circle(self, repeat: int=1):
            for i in range(repeat):
                p = partial(
                    CvUtil.circle,
                    center=CvUtil.Callback.get_position_interval,
                    radius=self.radius,
                    color=self.color,
                    thickness=self.lineType,
                    lineType=self.lineType
                )
                self._draw_queue.append(p)
        
        def rectangle(self, repeat: int=1):
            for i in range(repeat):
                p = partial(
                    CvUtil.rectangle,
                    pt1=CvUtil.Callback.get_position_interval,
                    pt2=CvUtil.Callback.get_position_interval,
                    color=self.color,
                    thickness=self.thickness,
                    lineType=self.lineType,
                    rotation=Interval[float](-180, 180)
                )
                self._draw_queue.append(p)
        
        def text(self, repeat: int=1):
            for i in range(repeat):
                p = partial(
                    CvUtil.text,
                    text=self.text_gen,
                    org=CvUtil.Callback.get_position_interval,
                    color=self.color,
                    thickness=self.thickness,
                    lineType=self.lineType
                )
                self._draw_queue.append(p)

        def draw(self) -> np.ndarray:
            img = self.img.copy()
            for p in self._draw_queue:
                img = p(img)
            return img
        
        def draw_batch(self, batch_size: int=10) -> list[np.ndarray]:
            imgs: list[np.ndarray] = []
            for i in range(batch_size):
                imgs.append(self.draw())
            return imgs

        def draw_batch_parallel(self, n_jobs=2, batch_size: int=10) -> list[np.ndarray]:
            return Parallel(n_jobs=n_jobs)(delayed(self.draw)() for i in range(batch_size))
        
        @staticmethod
        def debug():
            worker = Worker()
            worker.circle(repeat=3)
            worker.rectangle(repeat=4)
            worker.text(repeat=5)

            batch_size = 100; n_jobs = 4
            t0 = time.time()
            imgs0 = worker.draw_batch(batch_size=batch_size)
            t1 = time.time()
            imgs1 = worker.draw_batch_parallel(n_jobs=n_jobs, batch_size=batch_size)
            t2 = time.time()
            print(f"non-parallel: {t1-t0}, parallel: {t2-t1}")
            # for result in imgs1:
            #     cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            #     cv2.resizeWindow('result', 500, 500)
            #     cv2.imshow('result', result)
            #     cv2.waitKey()
            #     cv2.destroyAllWindows()
    
    Worker.debug()
