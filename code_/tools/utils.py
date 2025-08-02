import time
import functools
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    

def timeit(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start = time.monotonic()
        res = func(*args, **kwargs)
        end = time.monotonic()
        print(f"------------------Total time for {func.__name__}: {end - start:.3f} s ------------------")
        return res
    return wrapped