from __future__ import annotations
import time

class TimeProbe:
    def __init__(self, start: float, end: float=None):
        self.start = start
        self.end = end

    @property
    def timeElapsed(self) -> float:
        return time.time() - self.start
    
    @property
    def measurement(self) -> float | None:
        if self.end is None:
            return None
        else:
            return self.end - self.start

class TimeProber:
    def __init__(self):
        self.probeNames: list[str] = []
        self.probes: list[TimeProbe] = []
        self._pendingName: str = None

    def __call__(self, name: str) -> TimeProber:
        self._pendingName = name
        return self

    def __enter__(self) -> TimeProbe:
        probe = TimeProbe(start=time.time())
        assert self._pendingName not in self.probeNames
        self.probeNames.append(self._pendingName)
        self.probes.append(probe)
        self._pendingName = None
        return probe

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        name = self.probeNames.pop()
        probe = self.probes.pop()
        probe.end = time.time()
        print(f"{name}: {probe.measurement}")

class DebugTimer:
    probe: TimeProber = TimeProber()

    @classmethod
    def debug(cls):
        with DebugTimer.probe('Flag0') as t0:
            for i in range(3):
                with DebugTimer.probe('Flag1') as t1:
                    time.sleep(1)
