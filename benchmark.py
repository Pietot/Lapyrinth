""" __sumary__ """


import timeit
import threading as thrd


from typing import Callable
from matplotlib import pyplot as pltu-

from maze import Maze


def get_time(maze: Maze, algorithm: Callable[[Maze], Maze], iterations: int = 1) -> float:
    """ __sumary__ """
    return timeit.timeit(lambda: algorithm(maze), number=iterations)
