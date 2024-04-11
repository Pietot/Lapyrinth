"""Maze Maker Solver"""


# By Pietot
# Discord : Piétôt#1754 | pietot
# Start : 15/03/2024 at 15h30 FR
# End : 19/03/2024 at 16h30 FR


# v1.1 :
# Start : 19/03/2024 at 16h50 FR
# End : 20/03/2024 at 22h30 FR
# Changelogs : Added numpy loop, type checking, rasing error, better code, better logic and more

# v1.2 :
# Start : 21/03/2024 at 11h00 FR
# End : 04/03/2024 at h FR
# Changelogs : Minor changes, just improved some code by optimizing it and/or refactoring it
#              and implemented better logic


from typing import Any, Generator

import sys

import random as rdm
import numpy as np


sys.setrecursionlimit(10000)


class Maze:
    """ Maze class
    """

    def __init__(self, shape: Any | int | tuple[int, int] = 5, raise_error: bool = True) -> None:
        shape = verify_shape(shape, raise_error)
        self.maze = np.zeros(shape, dtype=object)
        self.algorithm: None | str = None
        self.is_complexe = False
        self.sculpt_grid()

    def __str__(self) -> str:
        maze = [['# ' if value in (0, 1) else '  ' for value in row]
                for row in self.maze]
        return '\n'.join(''.join(row) for row in maze)

    def __repr__(self) -> str:
        return np.array2string(self.maze, separator=' ')

    def __iter__(self) -> Generator[tuple[tuple[int, ...], np.uint16], None, None]:
        for index, value in np.ndenumerate(self.maze):
            yield index, value

    def sculpt_grid(self) -> 'Maze':
        """ Create the grid , 0 is for pillars,
            1 for breakable walls and other for paths
        """
        tile_value = 2
        for index, _ in self:
            # If we are not at the edges
            if not (index[0] in (0, self.maze.shape[0] - 1)
                    or index[1] in (0, self.maze.shape[1] - 1)):
                # If coordinates are odd
                if (index[0] % 2, index[1] % 2) == (1, 1):
                    self.maze[index[0]][index[1]] = tile_value
                    tile_value += 1
                # If wall are not intersections
                elif (index[0] % 2, index[1] % 2) != (0, 0):
                    self.maze[index] = 1
        self.was_scuplted = True
        return self

    def kruskal(self, breakable_walls: None | list[tuple[int, int]] = None) -> 'Maze':
        """_summary_

        Args:
            breakable_walls (None | list[tuple[int, int]], optional): _description_.
            Defaults to None.
            raise_error (bool, optional): _description_. Defaults to True.

        Returns:
            Maze: _description_
        """
        if breakable_walls is None:
            breakable_walls = self.get_breakable_walls()
        if not breakable_walls:
            # We set the entry and the exit
            self.maze[1][0], self.maze[self.maze.shape[0] - 2][self.maze.shape[1]-1] = (2, 2)
            self.algorithm = "Merge Path"
            return self
        coordinates = breakable_walls[0]
        if coordinates[0] % 2 == 0:
            upper_value = self.maze[coordinates[0]-1, coordinates[1]]
            bottom_value = self.maze[coordinates[0]+1, coordinates[1]]
            values = (upper_value, bottom_value)
        else:
            left_value = self.maze[coordinates[0], coordinates[1]-1]
            right_value = self.maze[coordinates[0], coordinates[1]+1]
            values = (left_value, right_value)
        breakable_walls.remove(coordinates)
        # If the values are the same, we don't destroy the wall or we will create a loop
        if values[0] == values[1]:
            return self.kruskal(breakable_walls)
        self.destroy_wall(coordinates, values)
        return self.kruskal(breakable_walls)

    def get_breakable_walls(self) -> list[tuple[int, int]]:
        """ Get all breakable walls coordinates

        Returns:
            list[tuple[int, int]]: List of all breakable walls coordinates
        """
        coordinates: list[tuple[int, int]] = []
        for index, value in self:
            if value == 1 or isinstance(value, tuple) and value[0] == 1:
                coordinates.append((index[0], index[1]))
        rdm.shuffle(coordinates)
        return coordinates

    def make_complex_maze(self, probability: int | float = 0.2) -> 'Maze':
        """ Make the maze more complex by removing some walls randomly

        Args:
            probability (int | float, optional): Probability of removing a wall. Defaults to 0.2.
        """
        # Force the probability to be between 0 and 1
        probability = max(0, min(1, probability))
        for index, value in self:
            if value == 1 and 0 < rdm.uniform(0, 1) <= probability:
                self.maze[index] = 2
        self.is_complexe = True
        return self

    def destroy_wall(self, wall_coordinate: tuple[int, int], values: tuple[int, int]) -> 'Maze':
        """ Destroy a wall and merging the values

        Args:
            wall_coordinate (tuple[int, int]): The wall coordinates
            values (tuple[int, int]): The values to merge
        """
        selected_value = rdm.choice(values)
        value_to_replace = values[0] if selected_value == values[1] else values[1]
        for index, value in self:
            if value == value_to_replace:
                self.maze[index] = selected_value
        self.maze[wall_coordinate[0], wall_coordinate[1]] = selected_value
        return self


def verify_shape(shape: Any | tuple[Any, ...], raise_error: bool) -> tuple[int, int]:
    """ Verify the shape of the maze

    Args:
        shape (Any | tuple[Any, ...]): Shape of the maze.
    """
    if isinstance(shape, int):
        if not (shape < 5 or shape % 2 == 0):
            return shape, shape
        if raise_error:
            raise ValueError("Shape must be greater than 5 and odd")
        return shape-1, shape-1
    if isinstance(shape, tuple):
        if len(shape) == 2 and all(isinstance(i, int) for i in shape if i > 5 and i % 2 == 1):
            return shape
        if raise_error:
            raise ValueError("Shape must be a tuple of 2 integer greater than 5 and odd")
        return ((shape[0] - 1) if shape[0] % 2 == 0 else shape[0],
                (shape[1] - 1) if shape[1] % 2 == 0 else shape[1])
    if raise_error:
        raise ValueError("Shape must be an int or a tuple[int, int]")
    return 5, 5


x = Maze(11)
print(x.kruskal())
print(repr(x))
