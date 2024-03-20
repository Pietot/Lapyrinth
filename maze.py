"""Maze Maker Solver"""


# By Pietot
# Discord : Piétôt#1754 | pietot
# Start : 15/03/2024 at 15h30 FR
# End : 19/03/2024 at 16h30 FR


# v1.1 :
# Start : 19/03/2024 at 16h50 FR
# End : 20/03/2024 at 22h30 FR
# Changelogs : Added numpy loop, type checking, rasing error, better code, better logic and more


from typing import Any, Generator

import sys

import random as rdm
import numpy as np


sys.setrecursionlimit(10000)


class Maze:
    """ Maze class
    """

    def __init__(self, shape: Any | tuple[Any, ...] = 5) -> None:
        shape = verify_shape(shape)
        self.maze = np.zeros(shape, dtype=np.uint16)
        self.algorithm = None
        self.is_complexe = False
        self.is_perfect = False
        self.was_scuplted = False

    def __str__(self) -> str:
        maze = [['# ' if value in (0, 1) else '  ' for value in row]
                for row in self.maze]
        return '\n'.join(''.join(row) for row in maze)

    def __iter__(self) -> Generator[tuple[tuple[int, ...], np.uint16], None, None]:
        for index, value in np.ndenumerate(self.maze):
            yield index, value

    def sculpt_grid(self) -> 'Maze':
        """ Create the grid with entry and exit, 0 is for pillars,
            1 for breakable walls and other for paths
        """
        tile_value = 2
        for index, _ in self:
            # If we are not at the edges
            if not (index[0] in (0, self.maze.shape[0] - 1) or
                    index[1] in (0, self.maze.shape[1] - 1)):
                # If coordinates are odd
                if (index[0] % 2, index[1] % 2) == (1, 1):
                    self.maze[index[0]
                              ][index[1]] = tile_value
                    tile_value += 1
                # If wall are not intersections
                elif (index[0] % 2, index[1] % 2) != (0, 0):
                    self.maze[index[0]][index[1]] = 1
        self.maze[1][0], self.maze[self.maze.shape[0] - 2][self.maze.shape[1]-1] = (2, tile_value-1)
        self.was_scuplted = True
        return self

    def merge_path(self, breakable_walls: None | list[tuple[int, int]] = None,
                   raise_error: bool = True) -> 'Maze':
        """_summary_

        Args:
            breakable_walls (None | list[tuple[int, int]], optional): 
            List of all breakable walls coordinates Defaults to None.
        """
        if 0 in (self.maze.shape[0] % 2, self.maze.shape[1] % 2):
            if raise_error:
                raise ValueError("For merge_path alogithm, shape[0] AND shape[1] must be odd")
            # Removing the penultimate line/column (not the last because it's the edge of the maze)
            if self.maze.shape[0] % 2 == 0:
                self.maze = np.delete(self.maze, -2, axis=0)
                # We replace the exit since we removed it by deleting the penultimate line
                self.maze[self.maze.shape[0] - 2][self.maze.shape[1]-1] = 2
            if self.maze.shape[1] % 2 == 0:
                self.maze = np.delete(self.maze, -2, axis=1)
        if not self.was_scuplted:
            self.sculpt_grid()
        if breakable_walls is None:
            breakable_walls = self.get_breakable_walls()
        values = (value for row in self.maze for value in row)
        if len(np.unique(self.maze)) == 3:
            return self
        coordinates = rdm.choice(breakable_walls)
        if coordinates[0] % 2 == 0:
            upper_value = self.maze[coordinates[0]-1, coordinates[1]]
            bottom_value = self.maze[coordinates[0]+1, coordinates[1]]
            values = (upper_value, bottom_value)
        else:
            left_value = self.maze[coordinates[0], coordinates[1]-1]
            right_value = self.maze[coordinates[0], coordinates[1]+1]
            values = (left_value, right_value)
        breakable_walls.remove(coordinates)
        if values[0] == values[1]:
            return self.merge_path(breakable_walls)
        self.destroy_wall(coordinates, values)
        return self.merge_path(breakable_walls)

    # def merge_path_iterative(self, breakable_walls: None | list[tuple[int, int]] = None,
    #                 raise_error: bool = True) -> 'Maze':
    #     if self.maze.shape[1] % 2 == 0:
    #         if raise_error:
    #             raise ValueError("For merge_path alogithm, shape[1] must be odd")
    #         # Removing the penultimate column (not the last because it's the edge of the maze)
    #         self.maze = np.delete(self.maze, -2, axis=1)
    #     if not self.was_scuplted:
    #         self.sculpt_grid()
    #     if breakable_walls is None:
    #         breakable_walls = self.get_breakable_walls()
    #     values = (value for row in self.maze for value in row)
    #     while len(np.unique(self.maze)) != 3:
    #         breakable_wall = rdm.choice(breakable_walls)
    #         if breakable_wall[0] % 2 == 0:
    #             upper_value = self.maze[breakable_wall[0]-1, breakable_wall[1]]
    #             bottom_value = self.maze[breakable_wall[0]+1, breakable_wall[1]]
    #             values = (upper_value, bottom_value)
    #         else:
    #             left_value = self.maze[breakable_wall[0], breakable_wall[1]-1]
    #             right_value = self.maze[breakable_wall[0], breakable_wall[1]+1]
    #             values = (left_value, right_value)
    #         if values[0] == values[1]:
    #             continue
    #         self.destroy_wall(breakable_wall, values)
    #     return self

    def get_breakable_walls(self) -> list[tuple[int, int]]:
        """ Get all breakable walls coordinates

        Returns:
            list[tuple[int, int]]: List of all breakable walls coordinates
        """
        coordinates: list[tuple[int, int]] = []
        for index, value in self:
            if value.item() == 1:  # type: ignore
                coordinates.append((index[0], index[1]))
        return coordinates

    def make_complex_maze(self, probability: int | float = 0.2) -> 'Maze':
        """ Make the maze more complex by removing some walls randomly

        Args:
            probability (int | float, optional): Probability of removing a wall. Defaults to 0.2.
        """
        # Force the probability to be between 0 and 1
        probability = max(0, min(1, probability))
        for index, value in self:
            if value.item() == 1 and 0 < rdm.uniform(0, 1) <= probability:  # type: ignore
                self.maze[index[0]][index[1]] = 2
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
            if value.item() == value_to_replace:  # type: ignore
                self.maze[index[0]][index[1]] = selected_value
        self.maze[wall_coordinate[0], wall_coordinate[1]] = selected_value
        return self


def verify_shape(shape: Any | tuple[Any, ...]) -> tuple[int, int]:
    """ Verify the shape of the maze

    Args:
        shape (Any | tuple[Any, ...]): Shape of the maze.
    """
    if isinstance(shape, int):
        if shape < 5:
            raise ValueError("Shape must greater than 5")
        return shape, shape
    if isinstance(shape, tuple):
        if len(shape) != 2:
            raise ValueError("Shape must be a tuple of 2 integer greater than 5")
        if not all(isinstance(i, int) for i in shape if i > 5):
            raise ValueError("Shape must be a tuple of 2 integer greater than 5")
        return shape
    raise ValueError("Shape must be an int or a tuple[int, int]")


print(Maze((12, 11)).sculpt_grid().merge_path(raise_error=False))
