""" A program capable of solving mazes with different path-finding algorithms. """


# By Pietot
# Discord : Piétôt#1754 | pietot
# Start : 16/05/2024 at 13h30 FR
# End : /05/2024 at h FR

# v1.0 :
# Start : 16/05/2024 at 13h30 FR
# End : 19/05/2024 at 23h15 FR
# Changelogs : Added the left hand rule pathfinder

# v1.0 :
# Start : 20/05/2024 at 13h50 FR
# End : 20/05/2024 at 13h50 FR
# Changelogs : Added the right hand rule pathfinder


import colorsys

import numpy as np

from PIL import Image, ImageDraw

from maze import Maze

import maze


class UnsolvableMaze(Exception):
    """ Exception class for unsolvable Mazes.

    Args:
        Exception:
    """

    def __init__(self, algorithm: str) -> None:
        self.message = f"{algorithm} can't solve this maze"
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
    cell_with_direction: dict[tuple[int, int], list[tuple[int, int]]] = {}
    direction = next(iter(direction_to_left))
    while current_cell != self.end:
        left_cell_col = current_cell[1] + direction_to_left[direction][1]
        left_cell_row = current_cell[0] + direction_to_left[direction][0]
        if self.maze[left_cell_row][left_cell_col] > 1:
            direction = rotate_90_counterclockwise(direction)
            update_cell_directions(cell_with_direction, current_cell, direction, "Left")
            current_cell = (left_cell_row, left_cell_col)
            continue
        front_cell_row = current_cell[0] + direction[0]
        front_cell_col = current_cell[1] + direction[1]
        if self.maze[front_cell_row][front_cell_col] > 1:
            update_cell_directions(cell_with_direction, current_cell, direction, "Left")
            current_cell = (front_cell_row, front_cell_col)
        else:
            update_cell_directions(cell_with_direction, current_cell, direction, "Left")
            direction = rotate_90_clockwise(direction)
    path: list[tuple[int, int]] = []
    current_cell = self.start
    while current_cell != self.end:
        path.append(current_cell)
        current_cell = (current_cell[0] + cell_with_direction[current_cell][-1][0],
                        current_cell[1] + cell_with_direction[current_cell][-1][1])
    path.append(current_cell)
    return path


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
    cell_with_direction: dict[tuple[int, int], list[tuple[int, int]]] = {}
    direction = next(iter(direction_to_right))
    while current_cell != self.end:
        left_cell_col = current_cell[1] + direction_to_right[direction][1]
        left_cell_row = current_cell[0] + direction_to_right[direction][0]
        if self.maze[left_cell_row][left_cell_col] > 1:
            direction = rotate_90_clockwise(direction)
            update_cell_directions(cell_with_direction, current_cell, direction, "Right")
            current_cell = (left_cell_row, left_cell_col)
            continue
        front_cell_row = current_cell[0] + direction[0]
        front_cell_col = current_cell[1] + direction[1]
        if self.maze[front_cell_row][front_cell_col] > 1:
            update_cell_directions(cell_with_direction, current_cell, direction, "Right")
            current_cell = (front_cell_row, front_cell_col)
        else:
            update_cell_directions(cell_with_direction, current_cell, direction, "Right")
            direction = rotate_90_counterclockwise(direction)
    path: list[tuple[int, int]] = []
    current_cell = self.start
    while current_cell != self.end:
        path.append(current_cell)
        current_cell = (current_cell[0] + cell_with_direction[current_cell][-1][0],
                        current_cell[1] + cell_with_direction[current_cell][-1][1])
    path.append(current_cell)
    return path


def rotate_90_clockwise(direction: tuple[int, int]) -> tuple[int, int]:
    """ Rotate a direction 90 degrees clockwise.

    Args:
        direction (tuple[int, int]): The direction to rotate.

    Returns:
        tuple[int, int]: The rotated direction.
    """
    row, column = direction
    return (column, -row)


def rotate_90_counterclockwise(direction: tuple[int, int]) -> tuple[int, int]:
    """ Rotate a direction 90 degrees counterclockwise.

    Args:
        direction (tuple[int, int]): The direction to rotate.

    Returns:
        tuple[int, int]: The rotated direction.
    """
    row, column = direction
    return (-column, row)


def update_cell_directions(cell_with_direction: dict[tuple[int, int], list[tuple[int, int]]],
                           current_cell: tuple[int, int],
                           direction: tuple[int, int],
                           wall_follower_direction: str) -> None:
    """ Update the cell_with_direction dictionary with the current cell and direction.

    Args:
        cell_with_direction (dict[tuple[int, int], list[tuple[int, int]]]): _description_
        current_cell (tuple[int, int]): _description_
        direction (tuple[int, int]): _description_
        wall_follower_direction (str): _description_

    Raises:
        UnsolvableMaze: _description_
    """
    if cell_with_direction.get(current_cell):
        if direction in cell_with_direction[current_cell]:
            raise UnsolvableMaze(f"{wall_follower_direction} Hand Rule")
        cell_with_direction[current_cell].append(direction)
    else:
        cell_with_direction[current_cell] = [direction]


def generate_path(self: Maze, path: list[tuple[int, int]],
                  filename: str | None = None) -> None:
    """ Generate a maze image from a maze object. """
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
