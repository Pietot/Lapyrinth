""" A program capable of solving mazes with different path-finding algorithms. """


# By Pietot
# Discord : Piétôt#1754 | pietot
# Start : 16/05/2024 at 13h30 FR
# End : /05/2024 at h FR

# v1.0 :
# Start : 16/05/2024 at 13h30 FR
# End : 19/05/2024 at 23h15 FR
# Changelogs : Added the left hand rule pathfinder

# v1.1 :
# Start : 20/05/2024 at 13h50 FR
# End : 20/05/2024 at 13h50 FR
# Changelogs : Added the right hand rule pathfinder

# v1.2 :
# Start : 21/05/2024 at 21h35 FR
# End : 21/05/2024 at 23h00 FR
# Changelogs : Added random mouse pathfinder

# v1.3 :
# Start : 25/05/2024 at 19h10 FR
# End : 26/05/2024 at 17h30 FR
# Changelogs : Added Pledge pathfinder

# v1.4 :
# Start : 26/05/2024 at 22h30 FR
# End : 28/05/2024 at 00h35 FR
# Changelogs : Added Dead End Filler pathfinder

# v1.5 :
# Start : 28/05/2024 at 17h00 FR
# End : 28/05/2024 at 19h45 FR
# Changelogs : Added Depth First Search pathfinder


import colorsys

import random as rdm
import numpy as np

from PIL import Image, ImageDraw

from maze import Maze

import maze


class UnsolvableMaze(Exception):
    """ Exception class for unsolvable Mazes.

    Args:
        Exception: The base exception class.
    """

    def __init__(self, algorithm: str, error_message: str = "") -> None:
        if error_message:
            error_message = ":\n" + error_message
        self.message = f"{algorithm} cannot solve this maze in this configuration" + error_message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


def left_hand(self: Maze) -> list[tuple[int, int]]:
    """ Solve the maze with the left hand rule.

    It start by knowing if the left cell relative to the current direction is a wall or not.\n
    If it's not a wall, it will turn left, move forward and update the direction.\n
    Else, it checks if the front cell is a wall or not.\n
    If it's not a wall, it will move forward.\n
    Else, it will turn right and update the direction.\n

    To save the path, it will save the direction of the cell in a dictionary.

    Args:
        self (Maze): The maze object.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    direction_to_left: dict[tuple[int, int], tuple[int, int]] = {
        (0, 1): (-1, 0),
        (-1, 0): (0, -1),
        (0, -1): (1, 0),
        (1, 0): (0, 1)
    }
    current_cell: tuple[int, int] = self.start
    cell_with_directions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    direction = next(iter(direction_to_left))
    while current_cell != self.end:
        left_cell_col = current_cell[1] + direction_to_left[direction][1]
        left_cell_row = current_cell[0] + direction_to_left[direction][0]
        if self.maze[left_cell_row][left_cell_col] > 1:
            direction = turn_left(direction)
            update_cell_directions(cell_with_directions, current_cell, direction,
                                   algorithm="Left Hand Rule",
                                   error_message="Pathfinder is stuck in a loop.")
            current_cell = (left_cell_row, left_cell_col)
            continue
        front_cell_row = current_cell[0] + direction[0]
        front_cell_col = current_cell[1] + direction[1]
        if self.maze[front_cell_row][front_cell_col] > 1:
            update_cell_directions(cell_with_directions, current_cell, direction,
                                   algorithm="Left Hand Rule",
                                   error_message="Pathfinder is stuck in a loop.")
            current_cell = (front_cell_row, front_cell_col)
        else:
            update_cell_directions(cell_with_directions, current_cell, direction,
                                   algorithm="Left Hand Rule",
                                   error_message="Pathfinder is stuck in a loop.")
            direction = turn_right(direction)
    return directions_to_path(self, cell_with_directions)


def right_hand(self: Maze) -> list[tuple[int, int]]:
    """ Solve the maze with the right hand rule.

    It start by knowing if the right cell relative to the current direction is a wall or not.\n
    If it's not a wall, it will turn right, move forward and update the direction.\n
    Else, it checks if the front cell is a wall or not.\n
    If it's not a wall, it will move forward.\n
    Else, it will turn left and update the direction.\n

    To save the path, it will save the direction of the cell in a dictionary.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    direction_to_right: dict[tuple[int, int], tuple[int, int]] = {
        (0, 1): (1, 0),
        (1, 0): (0, -1),
        (0, -1): (-1, 0),
        (-1, 0): (0, 1)
    }
    current_cell: tuple[int, int] = self.start
    cell_with_directions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    direction = next(iter(direction_to_right))
    while current_cell != self.end:
        left_cell_col = current_cell[1] + direction_to_right[direction][1]
        left_cell_row = current_cell[0] + direction_to_right[direction][0]
        if self.maze[left_cell_row][left_cell_col] > 1:
            direction = turn_right(direction)
            update_cell_directions(cell_with_directions, current_cell, direction,
                                   algorithm="Right Hand Rule",
                                   error_message="Pathfinder is stuck in a loop.")
            current_cell = (left_cell_row, left_cell_col)
            continue
        front_cell_row = current_cell[0] + direction[0]
        front_cell_col = current_cell[1] + direction[1]
        if self.maze[front_cell_row][front_cell_col] > 1:
            update_cell_directions(cell_with_directions, current_cell, direction,
                                   algorithm="Right Hand Rule",
                                   error_message="Pathfinder is stuck in a loop.")
            current_cell = (front_cell_row, front_cell_col)
        else:
            update_cell_directions(cell_with_directions, current_cell, direction,
                                   algorithm="Right Hand Rule",
                                   error_message="Pathfinder is stuck in a loop.")
            direction = turn_left(direction)
    return directions_to_path(self, cell_with_directions)


def random_mouse(self: Maze) -> list[tuple[int, int]]:
    """ Solve the maze with the random mouse pathfinder.

    It will randomly choose a direction to move to until it reaches the end of the maze.\n
    For performance reasons, it will not choose the opposite direction until it's forced.\n

    Args:
        self (Maze): The maze object.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    current_cell = self.start
    path: list[tuple[int, int]] = [current_cell]
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    banned_direction = None

    while current_cell != self.end:
        neighbors = maze.get_neighbors(
            self, current_cell, directions=directions, return_visited=True)
        if banned_direction:
            neighbors = [neighbor for neighbor in neighbors if neighbor[1] != banned_direction]
        if not neighbors:
            neighbors = maze.get_neighbors(
                self, current_cell, directions=directions, return_visited=True)
        next_cell, direction = rdm.choice(neighbors)
        banned_direction = (-direction[0], -direction[1])
        path = update_path(path, next_cell)
        current_cell = next_cell
    return path


def pledge(self: Maze, following_direction: str) -> list[tuple[int, int]]:
    """ Solve the maze with the Pledge pathfinder.

    While the end is not reached, the pathfinder go straightforward.\n
    If it reaches a wall, it will walk along the left/right wall defined by following_direction\n
    For each turn, it will increase or decrease a counter (depending on the direction)\n
    If the counter reaches 0, it will go back to moving straightforward and the cycle restart.

    Args:
        self (Maze): The maze object.
        following_direction (str): The direction to follow the wall (left or right).

    Raises:
        ValueError: If the following_direction is not 'left' or 'right'.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    match following_direction:
        case "left":
            direction_to_turn: dict[tuple[int, int], tuple[int, int]] = {
                (0, 1): (-1, 0),
                (-1, 0): (0, -1),
                (0, -1): (1, 0),
                (1, 0): (0, 1)
            }
        case "right":
            direction_to_turn: dict[tuple[int, int], tuple[int, int]] = {
                (0, 1): (1, 0),
                (1, 0): (0, -1),
                (0, -1): (-1, 0),
                (-1, 0): (0, 1)
            }
        case _:
            raise ValueError("Invalid following_direction (must be 'left' or 'right')")
    current_cell: tuple[int, int] = self.start
    direction = next(iter(direction_to_turn))
    path: list[tuple[int, int]] = [current_cell]
    nb_rotations = 0
    while current_cell != self.end:
        front_neighbor = (current_cell[0] + direction[0],
                          current_cell[1] + direction[1])
        if nb_rotations == 0:
            if self.maze[front_neighbor[0]][front_neighbor[1]] > 1:
                current_cell = (current_cell[0] + direction[0],
                                current_cell[1] + direction[1])
                path = update_path(path, current_cell)
            else:
                direction = direction_to_turn[direction_to_turn[direction_to_turn[direction]]]
                nb_rotations += 1
        else:
            while nb_rotations != 0 and current_cell != self.end:
                front_neighbor = (current_cell[0] + direction[0],
                                  current_cell[1] + direction[1])
                side_to_follow = (current_cell[0] + direction_to_turn[direction][0],
                                  current_cell[1] + direction_to_turn[direction][1])
                if self.maze[side_to_follow[0]][side_to_follow[1]] > 1:
                    direction = direction_to_turn[direction]
                    nb_rotations -= 1
                    path = update_path(path, current_cell)
                    current_cell = (current_cell[0] + direction[0],
                                    current_cell[1] + direction[1])
                elif self.maze[front_neighbor[0]][front_neighbor[1]] < 2:
                    # Turning to one side equals turning to the other side 3 times
                    direction = direction_to_turn[direction_to_turn[direction_to_turn[direction]]]
                    nb_rotations += 1
                else:
                    current_cell = (current_cell[0] + direction[0],
                                    current_cell[1] + direction[1])
                    path = update_path(path, current_cell)

    return path


def dead_end_filler(self: Maze) -> list[tuple[int, int]]:
    """ Solve the maze with the Dead End Filler pathfinder.

    It start by converting all path cells to 2.\n
    Then if will recursively fill the dead ends (cells with only one neighbor)
    until there is no more dead ends.\n
    Finally, it will try get the path from the start to the end of the maze.\n
    If it finds multiple paths, in other words, a cell with two or more neighbor,
    it will raise an UnsolvableMaze exception.

    Args:
        self (Maze): The maze object.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    def fill() -> None:
        if dead_ends_indexes := get_dead_ends(self):
            rows, cols = zip(*dead_ends_indexes)
            self.maze[rows, cols] = 2
            fill()

    def get_path() -> list[tuple[int, int]]:
        current_cell: tuple[int, int] = self.start
        path: list[tuple[int, int]] = [self.start]
        while current_cell != self.end:
            neighbors = maze.get_neighbors(self, current_cell,
                                           directions=((-1, 0), (0, 1), (1, 0), (0, -1)))
            if len(neighbors) > 1:
                raise UnsolvableMaze("Dead End Filler",
                                     "Pathfinder found multiple paths but it can't select one.")
            self.maze[current_cell[0]][current_cell[1]] = 2
            current_cell = neighbors[0][0]
            path.append(current_cell)
        return path

    self.maze[self.maze > 1] = 3
    fill()
    return get_path()


def depth_first_search(self: Maze) -> list[tuple[int, int]]:
    """ Solve the maze with the Depth First Search pathfinder.
    
    It starts by converting all path cells to 3 (unvisited).
    Then it lists all the neighbors of the current cell, starting with the entry cell.
    It marks the current cell as visited.
    For each neighbor, it checks if it's the end, a dead end or a path.
    If the current cell is the end, it returns the path.
    If the current cell is a dead end, it backtracks to the last cell with multiple neighbors.
    If the current cell is a path, it will add the cell to the path,
    and recursively call the function with the neighbor as the current cell.
    This recursive call finishes when the end is reached or when all the cells has been visited.
    If the algorithm cannot find the end, it raises an UnsolvableMaze exception.

    Args:
        self (Maze): The maze object.

    Raises:
        UnsolvableMaze: If the algorithm cannot solve the maze due to the end not being reachable.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    self.maze[self.maze > 1] = 3

    def search(current_cell: tuple[int, int], path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        self.maze[current_cell[0]][current_cell[1]] = 2
        if current_cell == self.end:
            return path
        # Condition to optimize the search
        if current_cell == (self.end[0], self.end[1]-1):
            path.append(self.end)
            return path
        neighbors = maze.get_neighbors(self, current_cell,
                                       directions=((-1, 0), (0, 1), (1, 0), (0, -1)))
        for chosen_neighbor, _ in neighbors:
            if self.maze[chosen_neighbor[0]][chosen_neighbor[1]] == 3:
                path.append(chosen_neighbor)
                result = search(chosen_neighbor, path)
                if result:
                    return result
                path.pop()
        return []

    current_cell = self.start
    if path := search(current_cell, [current_cell]):
        return path
    raise UnsolvableMaze("Depth First Search", "End is not reachable.")


def turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    """ Rotate a direction 90 degrees clockwise.

    Args:
        direction (tuple[int, int]): The direction to rotate.

    Returns:
        tuple[int, int]: The rotated direction.
    """
    row, column = direction
    return (column, -row)


def turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    """ Rotate a direction 90 degrees counterclockwise.

    Args:
        direction (tuple[int, int]): The direction to rotate.

    Returns:
        tuple[int, int]: The rotated direction.
    """
    row, column = direction
    return (-column, row)


def update_path(path: list[tuple[int, int]], new_cell: tuple[int, int]) -> list[tuple[int, int]]:
    """ Update the path with the new cell.

    Args:
        path (list[tuple[int, int]]): The path to update.
        new_cell (tuple[int, int]): The new cell to add.

    Returns:
        list[tuple[int, int]]: The updated path.
    """
    if new_cell in path:
        path = path[:path.index(new_cell)+1]
    else:
        path.append(new_cell)
    return path


def update_cell_directions(cell_with_direction: dict[tuple[int, int], list[tuple[int, int]]],
                           current_cell: tuple[int, int],
                           direction: tuple[int, int],
                           algorithm: str,
                           error_message: str = "") -> None:
    """ Update the cell_with_direction dictionary with the current cell and direction.

    Args:
        cell_with_direction (dict[tuple[int, int], list[tuple[int, int]]]):
            The dictionary to update.
        current_cell (tuple[int, int]):  The current cell.
        direction (tuple[int, int]): The direction to add.
        algorithm (str): The algorithm used to solve the maze.
        error_message (str, optional): The error message to add if UnsolvableMaze is raised.
            Defaults to "".

    Raises:
        UnsolvableMaze: If the algorithm cannot solve the maze in the configuration given.
    """
    if cell_with_direction.get(current_cell):
        if direction in cell_with_direction[current_cell]:
            raise UnsolvableMaze(algorithm, error_message)
        cell_with_direction[current_cell].append(direction)
    else:
        cell_with_direction[current_cell] = [direction]


def get_dead_ends(self: Maze) -> list[tuple[int, int]]:
    """ Get the dead ends of the maze. Meaning the cells with only one neighbor.

    Args:
        self (Maze): The maze object.

    Returns:
        list[tuple[int, int]]: A list on indexes of the dead ends.
    """
    dead_ends: list[tuple[int, int]] = []
    start_neighbor = (1, 1)
    end_neighbor = (self.maze.shape[0] - 2, self.maze.shape[1] - 2)
    for index, value in np.ndenumerate(self.maze):
        # Skip the start and the end because they are not dead ends.
        # We also skip the neighbors of the start and the end
        # because we can't reach the start and the endas a neighbor.
        if (value < 2
            or index in (self.start, self.end, start_neighbor, end_neighbor)
                or value == 2):
            continue
        neighbors = maze.get_neighbors(self, (index[0], index[1]),
                                       directions=((-1, 0), (0, 1), (1, 0), (0, -1)))
        if len(neighbors) == 1:
            dead_ends.append((index[0], index[1]))
    return dead_ends


def directions_to_path(self: Maze,
                       cell_with_directions: dict[tuple[int, int],
                                                  list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """ Convert the cell_with_directions dictionary to a path.

    Args:
        self (Maze): The maze object.
        cell_with_directions (dict[tuple[int, int], list[tuple[int, int]]]):
            The dictionary to convert.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    path: list[tuple[int, int]] = []
    current_cell = self.start
    while current_cell != self.end:
        path.append(current_cell)
        current_cell = (current_cell[0] + cell_with_directions[current_cell][-1][0],
                        current_cell[1] + cell_with_directions[current_cell][-1][1])
    path.append(current_cell)
    return path


def generate_path(self: Maze, path: list[tuple[int, int]],
                  filename: str | None = None) -> None:
    """ Generate a maze image from a maze object.

    Args:
        self (Maze): The maze object.
        path (list[tuple[int, int]]): The path to draw.
        filename (str | None, optional): The filename. Defaults to None.
    """
    size = self.maze.shape
    filename = (filename + '.png' if filename
                else f'Maze_{size[0]}x{size[1]}_{self.algorithm}.png')
    cell_size = 50

    image = Image.new(
        "RGB", (size[0]*cell_size, size[1]*cell_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    path_length = len(path)

    def get_color(step: int, total_steps: int) -> tuple[int, int, int]:
        # Adjust hue to go from red (0) to yellow (1/6)
        hue = 1/6 * (step / total_steps)
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        return int(r * 255), int(g * 255), int(b * 255)

    def draw_path() -> None:
        for index, cell_value in np.ndenumerate(self.maze):
            x1 = index[1] * cell_size
            y1 = index[0] * cell_size
            x2 = (index[1] + 1) * cell_size
            y2 = (index[0] + 1) * cell_size

            if int(cell_value) in (0, 1):
                draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
            elif index in path:
                step = path.index(index)
                path_color = get_color(step, path_length)
                draw.rectangle((x1, y1, x2, y2), fill=path_color)
    draw_path()
    image.save(filename)
