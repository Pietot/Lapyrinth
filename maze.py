""" A program capable of creating mazes with many different algorithms
and solving them with different pathfinders """


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
# End : 04/03/2024
# Changelogs : Minor changes, just improved some code by optimizing it and/or refactoring it
#              and implemented better logic

# v1.3 :
# Start : 13/04/2024
# End : 04/04/2024 at 17h42 FR
# Changelogs : Added Depth First Search Algorithm


from typing import Any, Generator

import sys

import random as rdm
import numpy as np


sys.setrecursionlimit(10000)


class Maze:
    """ The Maze class
    """

    def __init__(self, shape: Any | int | tuple[int, int] = 5, raise_error: bool = True) -> None:
        shape = verify_shape(shape, raise_error)
        self.maze = np.zeros(shape, dtype=np.uint16)
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
        """ Creates the grid , 0 is for pillars,
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

    def kruskal(self, breakable_walls: list[tuple[int, int]] | None = None) -> 'Maze':
        """ Applies Kruskal's recursive algorithm to generate a maze.

        It starts by initializing each non-wall cell as unique value.
        For each breakable_walls (shuffled) it checks if the cells it connects are different.
        If they are, the program picks a value between them randomly
        and change all the other by the chosen value including the wall.
        If they are the same, the wall is not destroyed to avoid creating a loop.
        Finally, the wall is removed from the list of breakable walls.
        This process continues until the list if empty, resulting in a maze
        where each cell is connected to every other cell via a unique path without forming loops.

        Args:
            breakable_walls (None | list[tuple[int, int]], optional):
                A list of coordinates of all breakable walls. Defaults to None.

        Returns:
            Maze: The generated maze after applying Kruskal's algorithm.
        """
        if breakable_walls is None:
            breakable_walls = self.get_breakable_walls()
        if breakable_walls == []:
            # We set the entry and the exit
            self.maze[1][0], self.maze[self.maze.shape[0] - 2][self.maze.shape[1]-1] = (2, 2)
            self.algorithm = "Kruskal's algorithm"
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

    def depth_first_search(self, current_cell: tuple[int, int] = (0, 0),
                           visited: list[tuple[int, int]] | None = None) -> 'Maze':
        """ Applies the Depth First Search algorithm to generate a maze.

        It starts by initializing an empty list and choosing a random cell to start.
        If the neighbor cell has not been visited (or can't be visited),
        the wall between the two cells is destroyed and the neighbor cell becomes the current cell.
        This process continues until the current cell has no unvisited neighbors.
        The algorithm backtracks to the previous cell
        and repeats the process until all cells have been visited.

        Args:
            current_cell (tuple[int, int], optional):
                The current cell being visited. Defaults to (0, 0).
            visited (list[tuple[int, int]] | None, optional):
                A list of cells that have already been visited by the algorithm. Defaults to None.

        Returns:
            Maze: The generated maze after applying DFS algorithm.
        """
        if visited is None:
            visited = []
        if current_cell == (0, 0):
            current_cell = (rdm.randrange(1, self.maze.shape[0] - 2, 2),
                            rdm.randrange(1, self.maze.shape[1] - 2, 2))
        visited.append(current_cell)
        # North, East, South, West
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        rdm.shuffle(directions)
        for direction in directions:
            next_cell_row = current_cell[0] + direction[0]
            next_cell_column = current_cell[1] + direction[1]
            if not was_visited(self, next_cell_row, next_cell_column, visited):
                wall_coordinates = (current_cell[0] + direction[0] //
                                    2, current_cell[1] + direction[1] // 2)
                self.maze[wall_coordinates] = 2
                self.depth_first_search((next_cell_row, next_cell_column), visited)
        # We set the entry and the exit
        self.maze[1][0], self.maze[self.maze.shape[0] - 2][self.maze.shape[1]-1] = (2, 2)
        self.algorithm = "Depth First Search algorithm"
        return self

    def get_breakable_walls(self) -> list[tuple[int, int]]:
        """ Gets all breakable walls coordinates

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
        """ Makes the maze complex by removing some walls randomly

        Args:
            probability (int | float, optional):
            The probability of removing a wall. Defaults to 0.2.
        """
        # Force the probability to be between 0 and 1
        probability = max(0, min(1, probability))
        for index, value in self:
            if value == 1 and 0 < rdm.uniform(0, 1) <= probability:
                self.maze[index] = 2
        self.is_complexe = True
        return self

    def destroy_wall(self, wall_coordinate: tuple[int, int], values: tuple[int, int]) -> 'Maze':
        """ Destroys a wall and merging the values

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
    """ Verifies if shape of the maze if an int greater than 5 and odd
        or a tuple of 2 int greater than 5 and odd

    Args:
        shape (Any | tuple[Any, ...]): Shape of the maze.
        raise_error (bool):
            If the shape is invalid, raise an error if True, else return a valid or default shape.
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


def was_visited(self: Maze, row: int, column: int, visited: list[tuple[int, int]]) -> bool:
    """ Check if a cell has been visited.

    Args:
        self (Maze): The maze object.
        row (int): The row index of the cell.
        column (int): The column index of the cell.
        visited (list[tuple[int, int]]): A list of visited cells.

    Returns:
        bool: True if the cell has been visited, False otherwise.
    """
    if not 0 <= row < self.maze.shape[0] or not 0 <= column < self.maze.shape[1]:
        return True
    if self.maze[row][column] == 0:
        return True
    if (row, column) in visited:
        return True
    return False
