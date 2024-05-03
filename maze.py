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
# End : 14/04/2024 at 17h42 FR
# Changelogs : Added Depth First Search Algorithm

# v1.4 :
# Start : 15/04/2024 at 14h00 FR
# End : 15/04/2024 at 23h12 FR
# Changelogs : Added Prim's Algorithm

# v1.5 :
# Start : 16/04/2024 at 23h15 FR
# End : 17/04/2024 at 18h00 FR
# Changelogs : Added Hunt and Kill Algorithm

# v1.6 :
# Start : 24/04/2024 at 20h30 FR
# End : 25/04/2024 at 11h45 FR
# Changelogs : Added Binary Tree Algorithm

# v1.7 :
# Start : 21/04/2024 at 11h30 FR
# End : /04/2024 at h FR
# Changelogs : Added Recursive Division Algorithm

# v1.8 :
# Start : 18/04/2024 at 12h00 FR
# End : /04/2024 at h FR
# Changelogs : Added Eller's Algorithm


from typing import Any, Generator

import sys

import random as rdm
import numpy as np

from PIL import Image, ImageDraw


sys.setrecursionlimit(100000)


class Maze:
    """ The Maze class
    """

    def __init__(self, shape: Any | int | tuple[int, int] = 5, raise_error: bool = True) -> None:
        shape = verify_shape(shape, raise_error)
        self.maze = np.zeros(shape, dtype=np.uint)
        self.algorithm: None | str = None
        self.is_complexe = False
        self.is_empty = False
        self.have_value = False
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
        for index, _ in self:
            # If we are at the edges
            if (index[0] in (0, self.maze.shape[0] - 1)
                    or index[1] in (0, self.maze.shape[1] - 1)):
                continue
            # If coordinates are odd
            if (index[0] % 2, index[1] % 2) == (1, 1):
                self.maze[index] = 3
            # If wall are not intersections
            elif (index[0] % 2, index[1] % 2) != (0, 0):
                self.maze[index] = 1
        self.was_scuplted = True
        return self

    def set_value(self) -> 'Maze':
        """ Set a unique value to each cell

        Returns:
            Maze: The Maze object
        """
        value = 3
        for index, cell_value in self:
            if cell_value == 3:
                self.maze[index] = value
                value += 1
        return self

    def remove_walls(self) -> 'Maze':
        """ Remove all walls inside the maze

        Returns:
            Maze: The Maze object
        """
        for index, _ in self:
            # If we are not at the edges
            if not (index[0] in (0, self.maze.shape[0] - 1)
                    or index[1] in (0, self.maze.shape[1] - 1)):
                self.maze[index] = 2
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
        if not self.have_value:
            self.set_value()
            self.have_value = True
        if breakable_walls is None:
            breakable_walls = self.get_breakable_walls()
        if breakable_walls == []:
            # We set the entry and the exit
            self.maze[1][0], self.maze[self.maze.shape[0] -
                                       2][self.maze.shape[1]-1] = (2, 2)
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
        current_cell = (current_cell if current_cell != (0, 0)
                        else get_random_cell((self.maze.shape[0], self.maze.shape[1])))
        visited.append(current_cell)
        # North, East, South, West
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        rdm.shuffle(directions)
        for row, column in directions:
            next_cell = (current_cell[0] + row,
                         current_cell[1] + column)
            if not was_visited(self, (next_cell), visited):
                wall_coordinates = (current_cell[0] + row // 2,
                                    current_cell[1] + column // 2)
                self.maze[wall_coordinates] = 2
                self.depth_first_search(next_cell, visited)
        # We set the entry and the exit
        self.maze[1][0], self.maze[self.maze.shape[0] -
                                   2][self.maze.shape[1]-1] = (2, 2)
        self.algorithm = "Depth First Search algorithm"
        return self

    def prim(self, start: tuple[int, int] = (0, 0)) -> 'Maze':
        """Applies Prim's algorithm to generate a maze.

        It starts by selecting a starting cell, either specified in parameter or chosen randomly.
        Then it lists all its neighbors and adds them to the list of cells to explore.
        While there are neighbors to explore, it randomly selects one
        and if it was not explored, the wall between the two cells is destroyed.
        Finally it removes the neighbors from the list

        Args:
            start (tuple[int, int], optional):
                The starting cell coordinates.
                Defaults to (0, 0), meaning a random starting cell will be chosen within the maze.

        Returns:
            Maze: The generated maze after applying Prim's algorithm.
        """
        if not self.have_value:
            self.set_value()
            self.have_value = True
        neighbors: list[tuple[tuple[int, int], tuple[int, int]]] = []
        start = start if start != (0, 0) else get_random_cell(
            (self.maze.shape[0], self.maze.shape[1]))
        self.maze[start] = 2
        neighbors.extend(get_neighbors(self, start))
        while neighbors:
            neighbor, direction = rdm.choice(neighbors)
            # Avoid overlapping, maybe this condition can be removed idk
            if self.maze[neighbor] != 2:
                self.maze[neighbor] = 2
                wall_coordinates = (neighbor[0] - direction[0] // 2,
                                    neighbor[1] - direction[1] // 2)
                self.maze[wall_coordinates] = 2
            neighbors.remove((neighbor, direction))
            neighbors.extend(get_neighbors(self, neighbor))
            neighbors = list(set(neighbors))
        # We set the entry and the exit
        self.maze[1][0], self.maze[self.maze.shape[0] -
                                   2][self.maze.shape[1]-1] = (2, 2)
        self.algorithm = "Prim's algorithm"
        return self

    def hunt_and_kill(self, start: tuple[int, int] = (0, 0)) -> 'Maze':
        """ Applies Hunt and Kill algorithm to generate a maze.

        It starts at a random cell and carves a path to a random unvisited neighbor ("kill" phase).
        If there are no unvisited neighbors,
        it scans the grid for an unvisited cell that is adjacent to a visited one ("hunt" phase).
        The process ends when the "hunt" phase fails to find any suitable cells.

        Args:
            start (tuple[int, int], optional):
            The starting cell for the algorithm.
            Defaults to (0, 0), which means a random cell will be chosen.

        Returns:
            self: The generated maze after applying Hunt and Kill algorithm.
        """
        if not self.have_value:
            self.set_value()
            self.have_value = True
        start = start if start != (0, 0) else get_random_cell(
            (self.maze.shape[0], self.maze.shape[1]))

        def hunt() -> None:
            for index, cell_value in self:
                if int(cell_value) not in (0, 1, 2):
                    neighbor, direction = get_connection(
                        self, (index[0], index[1]))
                    if neighbor == (0, 0):
                        continue
                    self.maze[neighbor] = 2
                    wall_coordinates = (neighbor[0] - direction[0] // 2,
                                        neighbor[1] - direction[1] // 2)
                    self.maze[wall_coordinates] = 2
                    self.hunt_and_kill((index[0], index[1]))

        def kill(cell: tuple[int, int]) -> None:
            self.maze[cell] = 2
            neighbors = get_neighbors(self, cell)
            if not neighbors:
                return hunt()
            neighbor, direction = rdm.choice(neighbors)
            self.maze[neighbor] = 2
            wall_coordinates = (neighbor[0] - direction[0] // 2,
                                neighbor[1] - direction[1] // 2)
            self.maze[wall_coordinates] = 2
            return kill(neighbor)

        kill(start)
        # We set the entry and the exit
        self.maze[1][0], self.maze[self.maze.shape[0] -
                                   2][self.maze.shape[1]-1] = (2, 2)
        self.algorithm = "Hunt and Kill algorithm"
        return self

    def eller(self, probabilty: float | None = None) -> 'Maze':
        probabilty = (min(0.01, max(1, probabilty))if probabilty
                      else round(rdm.uniform(0.01, 1), 2))
        for index, value in self:
            value = int(value)
            if value in (0, 1) or rdm.random() > probabilty:
                continue
            if index[1] == self.maze.shape[1] - 2:
                values = value, self.maze[index[1]-2]
                wall_coordinates = (index[0], index[1]-1)
                self.destroy_wall(wall_coordinates, values)
            else:
                values = value, self.maze[index[1]+2]
                wall_coordinates = (index[0], index[1]+1)
                self.destroy_wall(wall_coordinates, values)
        return self

    def recursive_division(self, start: tuple[int, int] = (1, 1),
                           end: tuple[int, int] | None = None) -> 'Maze':
        def divide_vertically(width: int, height: int) -> int:
            if width == height:
                return rdm.getrandbits(1)
            return width > height

        def divide(start: tuple[int, int], end: tuple[int, int], ban: tuple[int, int] = (0, 0)) -> None:
            height = end[0] - start[0]
            width = end[1] - start[1]
            if height <= 1 or width <= 1:
                return
            if divide_vertically(width, height):
                wall_column_index = [i for i in range(
                    start[1], end[1]+1) if i not in (start[1], ban[1], end[1]) and i % 2 == 0]
                if not wall_column_index:
                    print("No wall column index found")
                    return
                wall_column_index = rdm.choice(wall_column_index)
                self.maze[start[0]:end[0] + 1, wall_column_index] = 0
                entry = rdm.randint(start[0], end[0])
                entry_coordinate = (entry, wall_column_index)
                self.maze[entry][wall_column_index] = 2
                divide(start, (end[0], wall_column_index - 1), entry_coordinate)
                divide((start[0], wall_column_index + 1), end, entry_coordinate)
            else:
                wall_row_index = [i for i in range(
                    start[0], end[0]+1) if i not in (start[0], ban[0], end[0]) and i % 2 == 0]
                if not wall_row_index:
                    print("No wall row index found")
                    return
                wall_row_index = rdm.choice(wall_row_index)
                self.maze[wall_row_index, start[1]:end[1] + 1] = 0
                entry = rdm.randint(start[1], end[1])
                entry_coordinate = (wall_row_index, entry)
                self.maze[wall_row_index][entry] = 2
                divide(start, (wall_row_index - 1, end[1]), entry_coordinate)
                divide((wall_row_index + 1, start[1]), end, entry_coordinate)
        if end is None:
            end = (self.maze.shape[0]-2, self.maze.shape[1]-2)
        if not self.is_empty:
            self.remove_walls()
            self.is_empty = True
        divide(start, end)
        # We set the entry and the exit
        self.maze[1][0], self.maze[self.maze.shape[0] -
                                   2][self.maze.shape[1]-1] = (2, 2)
        self.algorithm = "Recursive division Algorithm"
        return self

    def binary_tree(self) -> 'Maze':
        """ Applies the Binary Tree algorithm to generate a maze.

        It starts by iterating over the maze and checking if the cell is a path.
        Then looks for neighbors corresponding to the biais.
        For example, here the biais is ((-2, 0), (0, -2)) for nortwest.
        So it will look for the neighbors at (-2, 0) and (0, -2).
        After choosing randomly a neighbor, it will destroy the wall between the two cells.

        For difficulty reasons, the biais is arbitrary set to Nortwest beacause others biais
        are very easy to solve. If you want to choose the bais randomly (or not),
        you can use the following code:

        def binary_tree(self, biais: tuple[tuple[int, int],
                                           tuple[int, int]] | None = None) -> 'Maze':
            # Nortwest, Northeast, Southwest, Southeast
            biais_choices = ((-2, 0), (0, -2),
                    ((-2, 0), (0, 2)),
                    ((2, 0), (0, -2)),
                    ((2, 0), (0, 2)))
            biais = rdm.choice(biais_choices)

        Returns:
            Maze: The generated maze after applying Binary Tree algorithm.
        """
        # Northwest
        biais = ((-2, 0), (0, -2))
        for index, cell_value in self:
            # If coordinates are odd
            if ((index[0] % 2, index[1] % 2) not in ((1, 1), (0, 0))
                    or int(cell_value) in (0, 1)):
                continue
            neighbors = get_neighbors(
                self, (index[0], index[1]), biais, return_visited=True)
            if neighbors:
                neighbor, direction = rdm.choice(neighbors)
                wall_coordinates = (neighbor[0] - direction[0] // 2,
                                    neighbor[1] - direction[1] // 2)
                self.maze[wall_coordinates] = 2

        # We set the entry and the exit
        self.maze[1][0], self.maze[self.maze.shape[0] -
                                   2][self.maze.shape[1]-1] = (2, 2)
        self.algorithm = "Binary Tree Algorithm"
        return self

    def get_breakable_walls(self) -> list[tuple[int, int]]:
        """ Gets all breakable walls coordinates

        Returns:
            list[tuple[int, int]]: List of all breakable walls coordinates
        """
        coordinates: list[tuple[int, int]] = []
        for index, cell_value in self:
            if cell_value == 1 or isinstance(cell_value, tuple) and cell_value[0] == 1:
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
        for index, cell_value in self:
            if cell_value == 1 and 0 < rdm.uniform(0, 1) <= probability:
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
        for index, cell_value in self:
            if cell_value == value_to_replace:
                self.maze[index] = selected_value
        self.maze[wall_coordinate[0], wall_coordinate[1]] = selected_value
        return self

    def generate_image(self, filename: str | None = None) -> None:
        """ Generate a maze image from a maze object. """
        size = self.maze.shape
        filename = filename if filename else f'Maze_{size[0]}x{size[1]}_{self.algorithm}.png'
        cell_size = 50
        wall_color = (0, 0, 0)
        start = (1, 0)
        end = (size[0] - 2, size[1] - 1)

        image = Image.new(
            "RGB", (size[0]*cell_size, size[1]*cell_size), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for index, cell_value in self:
            x1 = index[1] * cell_size
            y1 = index[0] * cell_size
            x2 = (index[1] + 1) * cell_size
            y2 = (index[0] + 1) * cell_size

            if index == start:
                draw.rectangle((x1, y1+1, x2, y2), fill=(0, 255, 0))
            elif index == end:
                draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0))
            elif int(cell_value) in (0, 1):
                draw.rectangle((x1, y1, x2, y2), fill=wall_color)

        image.save(filename)


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
            raise ValueError(
                "Shape must be a tuple of 2 integer greater than 5 and odd")
        return ((shape[0] - 1) if shape[0] % 2 == 0 else shape[0],
                (shape[1] - 1) if shape[1] % 2 == 0 else shape[1])
    if raise_error:
        raise ValueError("Shape must be an int or a tuple[int, int]")
    return 5, 5


def was_visited(self: Maze, cell: tuple[int, int], visited: list[tuple[int, int]]) -> bool:
    """ Check if a cell has been visited.

    Args:
        self (Maze): The maze object.
        cell (tuple[int, int]): The indexes of the cell.
        visited (list[tuple[int, int]]): A list of visited cells.

    Returns:
        bool: True if the cell has been visited, False otherwise.
    """
    if not 0 <= cell[0] < self.maze.shape[0] or not 0 <= cell[1] < self.maze.shape[1]:
        return True
    if self.maze[cell[0]][cell[1]] == 0:
        return True
    if (cell[0], cell[1]) in visited:
        return True
    return False


def get_neighbors(self: Maze,
                  cell: tuple[int, int],
                  directions: tuple[tuple[int, int], ...] | None = None,
                  return_visited: bool = False) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Returns a list of neighboring cells that are accessible from the given cell.

    Args:
        self (Maze): The maze object.
        cell (tuple[int, int]): The coordinates of the cell.

    Returns:
        list[tuple[tuple[int, int], tuple[int, int]]]:
            A list of tuples representing neighboring cells along
            with the direction from the given cell.
    """
    neighbors: list[tuple[tuple[int, int], tuple[int, int]]] = []
    # North, East, South, West
    directions = directions if directions else (
        (-2, 0), (0, 2), (2, 0), (0, -2))
    for direction in directions:
        neighbor = cell[0] + direction[0], cell[1] + direction[1]
        if 1 <= neighbor[0] < self.maze.shape[0] and 1 <= neighbor[1] < self.maze.shape[1]:
            if return_visited or self.maze[neighbor] != 2:
                neighbors.append((neighbor, direction))
    return neighbors


def get_random_cell(shape: tuple[int, int]) -> tuple[int, int]:
    """ This function generates a random cell within a given shape.

    Args:
        shape (tuple[int, int]): A tuple representing the dimensions of the shape.
        The first integer is the height and the second integer is the width.

    Returns:
        tuple[int, int]: A tuple representing the coordinates of the randomly generated cell.
    """
    return (rdm.randrange(1, shape[0] - 2, 2),
            rdm.randrange(1, shape[1] - 2, 2))


def get_connection(self: Maze, index: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """ This method is used to get a connections of an unvisited cell to a visited cell in the maze.

    Args:
        self (Maze): An instance of the Maze class.
        index (tuple[int, int]): A tuple containing the coordinates of the cell in the maze.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]:
        A tuple containing two tuples. The first tuple is the coordinates of the visited cell.
        The second tuple is the direction of the visited cell relative to the unvisited cell.
        If no neighbor is connected, returns ((0, 0), (0, 0)).
    """
    neighbors: list[tuple[tuple[int, int], tuple[int, int]]] = []
    # North, East, South, West
    directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
    for row, column in directions:
        neighbor = (index[0] + row,
                    index[1] + column)
        if not 0 <= neighbor[0] < self.maze.shape[0] or not 0 <= neighbor[1] < self.maze.shape[1]:
            continue
        if self.maze[neighbor] == 2:
            neighbors.append((neighbor, (row, column)))
    if not neighbors:
        return (0, 0), (0, 0)
    return rdm.choice(neighbors)
