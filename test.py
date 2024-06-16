import random
from maze import Maze
import maze
from recursive_maze import RecursiveMaze
from timeit import timeit
import pathfinders
import numpy as np

x = maze.load_object("maze2.pkl")
print(timeit(lambda: pathfinders.a_star(x), number=1))
