import random
from maze import Maze
import maze
from recursive_maze import RecursiveMaze
from timeit import timeit
import pathfinders
import numpy as np
import glob
import os


def rename_images(directory):
    path = r"C:\Users\Bapti\OneDrive\Bureau\VS Code\a"
    files = list(filter(os.path.isfile, glob.glob(path + "\*")))
    for i, filename in enumerate(files):
        new_name = f"img_{i+1}.png"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)


directory = r"C:\Users\Bapti\OneDrive\Bureau\VS Code\a"
rename_images(directory)
