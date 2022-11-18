from __future__ import annotations
from typing import Callable, Iterable
import numpy as np
import numpy.typing as npt
import cv2

class SimpleVisualizer:
    """Simple Visualizer
    Controls:
        a: Previous Frame
        q: Quit
        other keys: Next Frame

    Usage:
        >>> vis = SimpleVisualizer()
        >>> with vis.loop(images) as loop:
        ...    while not loop.done:
        ...        image = images[loop.index]
        ...        vis.show(image, title=f"Image {loop.index}")
    """
    def __init__(self):
        self.isOpen = False
        self.waitKeyCallback: Callable[[int], bool] = None
        
        # Loop Related
        self._loopIndex: int | None = None
        self.loop = SimpleVisualizer.Loop(self)

    def show(self, img: npt.NDArray[np.uint8], title='image'):
        winName = 'preview'
        if not self.isOpen:
            cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(winName, 500, 500)
            self.isOpen = True
        
        cv2.setWindowTitle(winName, title)
        cv2.imshow(winName, img)
        while True:
            k = cv2.waitKey(1)
            if k == -1:
                continue
            else:
                if self.waitKeyCallback is not None:
                    doBreak = self.waitKeyCallback(k)
                    if doBreak:
                        break
                    else:
                        continue
                else:
                    break
    
    def close(self):
        if self.isOpen:
            cv2.destroyAllWindows()

    class Loop:
        def __init__(self, vis: SimpleVisualizer):
            self._vis = vis
            self._iter: Iterable | None = None
        
        @property
        def index(self) -> int | None:
            return self._vis._loopIndex
        
        @index.setter
        def index(self, value: int | None):
            self._vis._loopIndex = value

        @property
        def done(self) -> bool:
            return self.index is None or self.index >= len(self._iter)
        
        @done.setter
        def done(self, value: bool):
            if value:
                self.index = None
            elif not value and self.index is None:
                raise Exception
            else:
                pass

        def __call__(self, iter: Iterable) -> SimpleVisualizer.Loop:
            self._iter = iter

            if self.index is None:
                self.index = 0 if len(iter) > 0 else None

                def waitKeyCallback(k: int) -> bool:
                    if k == ord('a'):
                        self.index -= 1
                    elif k == ord('q'):
                        self.index = len(iter)
                    else:
                        self.index += 1
                    return True
                
                if self._vis.waitKeyCallback is None:
                    self._vis.waitKeyCallback = waitKeyCallback
                else:
                    raise ValueError(f"{type(self._vis).__name__} already has waitKeyCallback defined.")
            else:
                # already initialized in a previous loop
                pass
            
            return self

        def __enter__(self) -> SimpleVisualizer.Loop:
            if self._iter is None:
                raise ValueError("Need to initialize loop first using __call__.")
            return self

        def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
            self._iter = None
            self.index = None
            cv2.destroyAllWindows()
