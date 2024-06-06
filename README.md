# Maze-Maker-Solver

![Static Badge](https://img.shields.io/badge/made_in-France-red?labelColor=blue)
![Static Badge](https://img.shields.io/badge/language-Python-f7d54d?labelColor=4771a4)

<p align="center">
    <img src="assets/icon.ico" alt="Icon" width="300" style="border-radius:50%"/>
</p>

This **Mazer Maker Solver** made entirely in Python is a program where you can generate maze with many different algorithm and solving them with different pathfinders. Nothing more, nothing less.

## Download

- 🟢 **[Latest version](https://github.com/Pietot/Maze-Maker-Solver/archive/refs/heads/main.zip)**

## Features

- Generate any maze of any size

- Choose different algorithms from 11 of them (and more with different parameters)

- Solve them with different pathfiner from 0 of them

- Generate an image of the maze generated

- Ability to benchmark all algorithms in one click

## How to use ?

To begin, download and uncompress the project <a href="https://github.com/Pietot/Maze-Maker-Solver/archive/refs/heads/main.zip">here</a> or clone it by using one of the following command:
```
https://github.com/Pietot/Maze-Maker-Solver.git
```
or
```
git@github.com:Pietot/Maze-Maker-Solver.git
```

Then, you need to install all dependencies by opening a CLI (command line interface) and write these lines

```
cd "{the path to the project directory}"
pip install -r requirements.txt
```

Finally, write these lines at the end of <a href="https://github.com/Pietot/Maze-Maker-Solver/blob/main/maze.py">maze.py</a>:

```python
maze = Maze(x, y) # or Maze(x) for a maze of x*x cells
maze.binary_tree() # Choose the algorithm you want
print(maze) # If you want to print the maze in the CLI
maze.generate_image() # If you want to generate a .png file of the maze
```

or write these lines in another python file in the same directory as <a href="https://github.com/Pietot/Maze-Maker-Solver/blob/main/maze.py">maze.py</a>

```python
from maze import Maze

maze = Maze(x, y) # or Maze(x) for a maze of x*x cells
maze.binary_tree() # Choose the algorithm you want below
print(maze) # If you want to print the maze in the CLI
maze.generate_image() # If you want to generate a .png file of the maze
```

> **Note**: Obviously, the larger the maze, the longer it will take to create and generate the image.

That's it. See, it's very simple. You can go with all of these algorithms:

- Kruskal

<img src="assets/kruskal.png" width="300" style="border:solid white 1px"/><br><br>

- Randomized Depth First Search

<img src="assets/dfs.png" width="300" style="border:solid white 1px"/><br><br>

- Prim

<img src="assets/prim.png" width="300" style="border:solid white 1px"/><br><br>

- Hunt and Kill

<img src="assets/hunt_and_kill.png" width="300" style="border:solid white 1px"/><br><br>

- Eller (may vary depending on parameters)

<img src="assets/eller.png" width="300" style="border:solid white 1px"/><br><br>

- Iterative Division

<img src="assets/recursive_division.png" width="300" style="border:solid white 1px"/><br><br>

- Binary Tree (may vary depending on parameters)

<img src="assets/binary_tree.png" width="300" style="border:solid white 1px"/><br><br>

- Sidewinder (may vary depending on parameters)

<img src="assets/sidewinder_0.2.png" width="300" style="border:solid white 1px"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="assets/sidewinder_0.7.png" width="300" style="border:solid white 1px"/><br><br>

- Growing Tree (may vary depending on parameters)

<img src="assets/growing_tree_newest.png" width="300" style="border:solid white 1px"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="assets/growing_tree_random.png" width="300" style="border:solid white 1px"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="assets/growing_tree_mixed.png" width="300" style="border:solid white 1px"/>
<br><br>

- Aldous-Broder

<img src="assets/aldous_broder.png" width="300" style="border:solid white 1px"/><br><br>

- Wilson

<img src="assets/wilson.png" width="300" style="border:solid white 1px"/><br><br>

## Some statistics...

Wonder which one is **faster**? Or which one is less **memory intensive**?

Well.. I already did it for you! So here you are:

<img src="assets/Maze generation time depending on its size.svg">
<img src="assets/Maze generation memory location depending on the size.svg">

> **Note**: For the generation time, I deliberately excluded Aldous-Broder and Wilson algorithms because they are truly random (""luck"" based in other words), so they are very inconsistent on a generation to another.

If you want the values of these graphs with Aldous-Broder and Wilson algorithms, watch this:

<img src="assets/time_complexity.png">
<a href="assets/time_complexity.csv">Download csv here</a><br><br>
<img src="assets/memory_complexity.png">
<a href="assets/memory_complexity.csv">Download csv here</a><br><br>

If you want to benchmark these algorithms yourself:

- Download the benchmark file <a href="https://github.com/Pietot/Maze-Maker-Solver/blob/main/benchmark_generation_algorithm.py">here</a> and put it in the same folder as <a href="https://github.com/Pietot/Maze-Maker-Solver/blob/main/maze.py">maze.py</a>

- Edit the benchmark file for your needs:<br>
  line 216/218 and 273/275 change the max size of the maze like you want and adapt the step in the for loop.

- Run the script

> **Note**: Keep in mind that the lower the step is, the more precision/values ​​you will have, but the longer the script will take, and vice versa if the step is high.<br><br>
> These values can change depending on the version of Python and your PC<br><br>
>For these benchmarks, I used Python 3.12.0 on a ryzen 5 3600, rtx 2060 with 2*8GB of RAM clocked at 3600Hz on Windows 10