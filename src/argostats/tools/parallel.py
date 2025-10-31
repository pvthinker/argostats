import multiprocessing as mp
import time
import datetime
import numpy as np


review = mp.context.process.active_children


def active():
    return [(t.pid, t.is_alive())
            for t in review()]


def execute_in_parallel(values, POOL):
    return POOL.map(f, values)


class Thread:
    def __init__(self):
        self.pool = mp.Pool(4)
        self.get_pid()

    def get_pid(self):
        self.pid = [t.pid
                    for t in self.pool._ctx.active_children()]


def get_pool(nprocs=None):
    if nprocs is None:
        nprocs = mp.cpu_count()//2
    return mp.Pool(nprocs)


def split_range_evenly(n, m):
    s, c = divmod(n, m)
    i0 = np.hstack([np.arange(c+1)*(s+1), (c*(s+1)+s)+np.arange(m-c)*s])
    return [slice(i0[k], i0[k+1])
            # if k < m-1 else
            # slice(i0[k], None)
            for k in range(m)]


class Chrono:
    def __init__(self):
        self.start()

    def start(self):
        self.tic = time.time()

    def stop(self):
        self.toc = time.time()
        return datetime.timedelta(seconds=int(self.toc-self.tic))

    def elapsed(self):
        elapsed = self.stop()
        print(f"Elapsed time: {str(elapsed)}")


def chrono(func):
    def wrapper(*args, **kwargs):
        c = Chrono()
        result = func(*args, **kwargs)
        c.elapsed()
        return result
    return wrapper
