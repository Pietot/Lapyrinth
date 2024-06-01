""" A program capable of creating mazes with many different recursive algorithms. """


# By Pietot
# Discord : Piétôt#1754 | pietot
# Start : 15/03/2024 at 15h30 FR
# End : 19/03/2024 at 16h30 FR


from typing import Generator

import random as rdm
import numpy as np

from PIL import Image, ImageDraw


class Maze:
    """ The Maze class\n
        0 is for pillars, 1 for breakable walls, 2 for visited cells and other for unvisited cells.
    """

    def __init__(self, *nb_cells_by_sides: int) -> None:
        nb_cells_by_sides = nb_cells_by_sides if nb_cells_by_sides else (5, 5)
        self.maze = np.zeros(cells_to_shape(*nb_cells_by_sides), dtype=np.uint)
        self.algorithm: None | str = None
        self.pathfinder: None | str = None
        self.have_value = False
        self.start = (1, 0)
        self.end = (self.maze.shape[0] - 2, self.maze.shape[1] - 1)
        self.sculpt_grid()

    def __str__(self) -> str:
        maze = [['# ' if value < 2 else '  ' for value in row]
                for row in self.maze]
        return '\n'.join(''.join(row) for row in maze)

    def __repr__(self) -> str:
        return np.array2string(self.maze, separator=' ')

    def __iter__(self) -> Generator[tuple[tuple[int, ...], np.uint16], None, None]:
        # The slice is using to avoid walls/edges. We only want to iterate over the cells.
        for index, value in np.ndenumerate(self.maze[1:-1:2, 1:-1:2]):
            yield (index[0]*2+1, index[1]*2+1), value

    def sculpt_grid(self) -> None:
        """ Creates the grid.
        """
        self.maze[1:-1:2, 2:-1:2] = 1
        self.maze[2:-1:2, 1:-1:2] = 1
        self.maze[1:-1:2, 1:-1:2] = 3
        self.was_scuplted = True

    def set_values(self) -> None:
        """ Set a unique value to each cell.
        """
        indices = np.where(self.maze == 3)
        self.maze[indices] = np.arange(3, 3 + len(indices[0]))

    def set_start_end(self) -> None:
        """ Set the entry and the exit of the maze.
        """
        self.maze[self.start],  self.maze[self.end] = (2, 2)

    def remove_walls(self) -> None:
        """ Remove all walls inside the maze.

        Returns:
            Maze: The Maze object.
        """
        self.maze[1:-1, 1:-1] = 3

    def recursive_kruskal(self, breakable_walls: list[tuple[int, int]] | None = None) -> None:
        """ Applies Kruskal's recursive algorithm to generate a maze.

        It starts by initializing each non-wall cell as unique value.\n
        For each breakable_walls (shuffled) it checks if the cells it connects are different.\n
        If they are, the program picks a value between them randomly
        and change all the other by the chosen value including the wall.\n
        If they are the same, the wall is not destroyed to avoid creating a loop.\n
        Finally, the wall is removed from the list of breakable walls.\n
        This process continues until the list if empty, resulting in a maze
        where each cell is connected to every other cell via a unique path without forming loops.

        Args:
            breakable_walls (None | list[tuple[int, int]], optional):
                A list of coordinates of all breakable walls. Defaults to None.
        """
        if not self.have_value:
            self.set_values()
            self.have_value = True
        if breakable_walls is None:
            breakable_walls = get_breakable_walls(self)
            rdm.shuffle(breakable_walls)
        if not breakable_walls:
            self.set_start_end()
            self.algorithm = "Recrusive Kruskal's algorithm"
            return None
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
            return self.recursive_kruskal(breakable_walls)
        self.merge_values(coordinates, values)
        return self.recursive_kruskal(breakable_walls)

    def recursive_backtracking(self, current_cell: tuple[int, int] | None = None) -> None:
        """ Applies the Recursive Backtracking algorithm to generate a maze.

        It starts by choosing a random cell to start and marking it as visited.\n
        Then it lists all the neighbors of the current cell and shuffles them.\n
        It loops over the unvisited neighbors.\n
        If the neighbor cell has not been visited,
        the wall between the two cells is destroyed and the neighbor becomes the current cell.\n
        This process continues until the current cell has no unvisited neighbors.\n
        Then the algorithm backtracks to the previous cell
        and repeats the process until all cells have been visited.

        Args:
            current_cell (tuple[int, int] | None, optional):
                The current cell being visited. Defaults to None.
        """
        current_cell = current_cell if current_cell else get_random_cell(
            (self.maze.shape[0], self.maze.shape[1]))
        self.maze[current_cell] = 2
        neighbors = get_neighbors(self, current_cell)
        if not neighbors:
            return
        rdm.shuffle(neighbors)
        for chosen_neighbor, direction in neighbors:
            if self.maze[chosen_neighbor] == 2:
                continue
            wall_coordinates = (current_cell[0] + direction[0] // 2,
                                current_cell[1] + direction[1] // 2)
            self.maze[wall_coordinates] = 2
            self.recursive_backtracking(chosen_neighbor)
        self.set_start_end()
        self.algorithm = "Recursive Backtracker algorithm"

    def recursive_hunt_and_kill(self, start: tuple[int, int] | None = None) -> None:
        """ Applies the Hunt and Kill algorithm recursively to generate a maze.

        It starts at a random cell and carves a path to a random unvisited neighbor (kill phase).\n
        If there are no unvisited neighbors,
        it scans the grid for an unvisited cell that is adjacent to a visited one (hunt phase).\n
        The process ends when the hunt phase fails to find any suitable cells.

        Args:
            start (tuple[int, int] | None, optional):
            The starting cell for the algorithm.\n
            Defaults to None, which means a random cell will be chosen.
        """
        start = start if start else get_random_cell(
            (self.maze.shape[0], self.maze.shape[1]))

        def hunt() -> None:
            for index, cell_value in self:
                if int(cell_value) > 2:
                    neighbor, direction = get_connection(
                        self, (index[0], index[1]))
                    if neighbor == (0, 0):
                        continue
                    self.maze[neighbor] = 2
                    wall_coordinates = (neighbor[0] - direction[0] // 2,
                                        neighbor[1] - direction[1] // 2)
                    self.maze[wall_coordinates] = 2
                    self.recursive_hunt_and_kill((index[0], index[1]))

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
        self.set_start_end()
        self.algorithm = "Recursive Hunt and Kill algorithm"

    def recursive_division(self) -> None:
        """ Applies the Recursive division algorithm to generate a maze.

        It starts by dividing the maze into two parts, either horizontally or vertically.\n
        Then it creates a wall in the middle of the division.\n
        After that, it creates a carve into the wall.\n
        This process continues recursively until the maze is fully divided.\n
        """
        def divide_vertically(width: int, height: int) -> int:
            return width > height if width != height else rdm.getrandbits(1)

        def divide(start: tuple[int, int], end: tuple[int, int],
                   ban: tuple[int, int] = (0, 0)) -> None:
            height = end[0] - start[0]
            width = end[1] - start[1]
            if height <= 1 or width <= 1:
                return
            if divide_vertically(width, height):
                wall_column_index = [i for i in range(
                    start[1], end[1]+1) if i not in (start[1], ban[1], end[1]) and i % 2 == 0]
                wall_column_index = rdm.choice(wall_column_index)
                self.maze[start[0]:end[0] + 1, wall_column_index] = 0
                entries = [i for i in range(start[0], end[0]+1) if i % 2 == 1]
                entry = rdm.choice(entries)
                entry_coordinate = (entry, wall_column_index)
                self.maze[entry, wall_column_index] = 2
                divide(start, (end[0], wall_column_index - 1), entry_coordinate)
                divide((start[0], wall_column_index + 1), end, entry_coordinate)
            else:
                wall_row_index = [i for i in range(
                    start[0], end[0]+1) if i not in (start[0], ban[0], end[0]) and i % 2 == 0]
                wall_row_index = rdm.choice(wall_row_index)
                self.maze[wall_row_index, start[1]:end[1] + 1] = 0
                entries = [i for i in range(start[1], end[1]+1) if i % 2 == 1]
                entry = rdm.choice(entries)
                entry_coordinate = (wall_row_index, entry)
                self.maze[wall_row_index, entry] = 2
                divide(start, (wall_row_index - 1, end[1]), entry_coordinate)
                divide((wall_row_index + 1, start[1]), end, entry_coordinate)
        self.remove_walls()
        start = (1, 1)
        end = (self.maze.shape[0]-2, self.maze.shape[1]-2)
        divide(start, end)
        self.set_start_end()
        self.algorithm = "Recursive division algorithm"

    def merge_values(self, wall_coordinate: tuple[int, int] | list[int],
                     values: tuple[int, int]) -> None:
        """ Destroys a wall and merging the values.

        Args:
            wall_coordinate (tuple[int, int] | list[int]): The wall coordinates.
            values (tuple[int, int]): The values to merge.
        """
        selected_value = values[0]
        value_to_replace = values[1]
        self.maze[self.maze == value_to_replace] = selected_value
        self.maze[wall_coordinate] = selected_value

    def make_imperfect_maze(self, mode: tuple[str, int | float]) -> None:
        """ Make the maze more complex by removing some walls randomly.

        Args:
            mode (tuple[str, int | float]): The mode to remove walls.
            The first element is the mode ('number' or 'probability').
            The second element is the number of walls to remove or the probability to remove a wall.
        """
        breakable_walls_coordinates = get_breakable_walls(self)
        if float(mode[1]) == 0.0:
            raise ValueError("The number of walls to remove or the probab must be greater than 0")
        if mode[0] == 'number':
            # Force the number to be between 0 and the number of breakable walls
            number = max(0, min(int(mode[1]), len(breakable_walls_coordinates)))
            for coordinates in rdm.sample(breakable_walls_coordinates, number):
                self.maze[coordinates] = 2
        elif mode[0] == 'probability':
            # Force the probability to be between 0 and 1
            probability = max(0, min(1, mode[1]))
            for coordinates in breakable_walls_coordinates:
                if 0 < rdm.uniform(0, 1) <= probability:
                    self.maze[coordinates] = 2
        else:
            raise ValueError("mode must be \"probability\" or \"number\"")

    def generate_image(self, filename: str | None = None) -> None:
        """ Generate a maze image from a maze object. """
        size = self.maze.shape
        filename = (filename + '.png' if filename
                    else f'Maze_{size[0]//2}x{size[1]//2}_{self.algorithm}.png')
        cell_size = 50
        wall_color = (0, 0, 0)

        image = Image.new(
            "RGB", (size[0]*cell_size, size[1]*cell_size), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for index, cell_value in np.ndenumerate(self.maze):
            x1 = index[1] * cell_size
            y1 = index[0] * cell_size
            x2 = (index[1] + 1) * cell_size
            y2 = (index[0] + 1) * cell_size

            if index == self.start:
                draw.rectangle((x1, y1+1, x2, y2), fill=(0, 255, 0))
            elif index == self.end:
                draw.rectangle((x1, y1+1, x2, y2), fill=(255, 0, 0))
            elif int(cell_value) < 2:
                draw.rectangle((x1, y1, x2, y2), fill=wall_color)

        image.save(filename)


def cells_to_shape(*nb_cells_by_side: int) -> tuple[int, int]:
    """ Convert the number of cells of each dimension (height, width) to the shape of the maze.

    Raises:
        ValueError: nb_cells_by_side must be an one or two int greater or equal to 2.

    Returns:
        tuple[int, int]: The shape of the maze.
    """
    if len(nb_cells_by_side) == 1 and nb_cells_by_side[0] >= 2:
        shape = (nb_cells_by_side[0]*2 + 1, nb_cells_by_side[0]*2 + 1)
        return shape
    if len(nb_cells_by_side) == 2 and all(cells >= 2 for cells in nb_cells_by_side):
        shape = (nb_cells_by_side[0]*2 + 1, nb_cells_by_side[1]*2 + 1)
        return shape
    raise ValueError("nb_cells_by_side must be an one or two int greater or equal to 2")


def get_breakable_walls(self: Maze) -> list[tuple[int, int]]:
    """ Gets all breakable walls coordinates.

    Returns:
        list[list[int, int]]: List of all breakable walls coordinates.
    """
    return [tuple(coord) for coord in np.argwhere(self.maze == 1).tolist()]


def get_unvisited_cells(self: Maze) -> list[list[int]]:
    """ Gets all unvisited cells coordinates.

    Returns:
        list[tuple[int, int]]: List of all unvisited cells coordinates.
    """
    return np.argwhere(self.maze > 2).tolist()


def get_neighbors(self: Maze,
                  cell: tuple[int, int],
                  directions: tuple[tuple[int, int], ...] | None = None,
                  return_visited: bool = False) -> list[tuple[tuple[int,
                                                                    int], tuple[int, int]]]:
    """Returns a list of neighboring cells that are accessible from the given cell.

    Args:
        self (Maze): The maze object.
        cell (tuple[int, int]): The coordinates of the cell.
        directions (tuple[tuple[int, int], ...], optional): The directions to check.
        Defaults to None.
        return_visited (bool): If we want to return visited neighbors.
        Defaults to False.

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
            if (return_visited and self.maze[neighbor] > 1) or self.maze[neighbor] > 2:
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
        A tuple containing two tuples.\n
        The first tuple is the coordinates of the visited cell.\n
        The second tuple is the direction of the visited cell relative to the unvisited cell.\n
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
    return rdm.choice(neighbors) if neighbors else ((0, 0), (0, 0))