# Maze-Maker-Solver

# Calendar

![Static Badge](https://img.shields.io/badge/made_in-France-red?labelColor=blue)
![Static Badge](https://img.shields.io/badge/language-Python-f7d54d?labelColor=4771a4)

<p align="center">
    <img src="assets/icon.ico" alt="Icon" width="300" style="border-radius:50%"/>
</p>

This **Mazer Maker Solver** made entirely in Python is a program where you can generate maze with many different algorithm and solving them with different pathfinders. Nothing more, nothing less.

## Download

- 🟢 **[Latest version](https://github.com/Pietot/Maze-Maker-Solver/blob/main/maze.py)**

## Feature

- Generate any maze of any size

- Choose different algorithms from 11 of them (and more with different parameters)

- Solve them with different pathfiner from 0 of them

- Generate an image of the maze generated

- Ability to benchmark all algorithms in one click

## How to use ?

To begin, write these lines at the end of :

```python 
maze = Maze(x, y) # or Maze(x) for a maze of x*x cells
maze.binary_tree()
print(maze)
maze.generate_image() # optional
```
