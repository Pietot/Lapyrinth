""" A program capable of solving mazes with different pathfinders """


# By Pietot
# Discord : Piétôt#1754 | pietot
# Start : 16/05/2024 at 13h30 FR
# End : /05/2024 at h FR

# v1.0 :
# Start : 16/05/2024 at 13h30 FR
# End : /05/2024 at h FR
# Changelogs : Added the left hand rule pathfinder


import random as rdm
import numpy as np

from PIL import Image, ImageDraw

from maze import Maze

import maze


def left_hand(self: Maze) -> list[tuple[int, int]]:
    """ Solve the maze with the left hand rule """
    direction_to_left: dict[tuple[int, int], tuple[int, int]] = {
        (0, 1): (-1, 0),
        (-1, 0): (0, -1),
        (0, -1): (1, 0),
        (1, 0): (0, 1)
    }
    current_cell: tuple[int, int] = self.start
    path: list[tuple[int, int]] = []
    direction = next(iter(direction_to_left))
    while current_cell != self.end:
        path.append(current_cell)
        left_cell_col = current_cell[1] + direction_to_left[direction][1]
        left_cell_row = current_cell[0] + direction_to_left[direction][0]
        if self.maze[left_cell_row][left_cell_col] not in (0, 1):
            direction = rotate_90_counterclockwise(direction)
            current_cell = (left_cell_row, left_cell_col)
            continue
        front_cell_row = current_cell[0] + direction[0]
        front_cell_col = current_cell[1] + direction[1]
        if self.maze[front_cell_row][front_cell_col] in (0, 1):
            direction = rotate_90_clockwise(direction)
        else:
            current_cell = (front_cell_row, front_cell_col)
    return list(dict.fromkeys(path))


def rotate_90_clockwise(direction: tuple[int, int]) -> tuple[int, int]:
    row, column = direction
    return (column, -row)


def rotate_90_counterclockwise(direction: tuple[int, int]) -> tuple[int, int]:
    row, column = direction
    return (-column, row)


def generate_path(self: Maze, path: list[tuple[int, int]],
                  filename: str | None = None) -> None:
    """ Generate a maze image from a maze object. """
    size = self.maze.shape
    filename = (filename + '.png' if filename
                else f'Maze_{size[0]}x{size[1]}_{self.algorithm}.png')
    cell_size = 50
    wall_color = (0, 0, 0)
    path_color = "cyan"

    image = Image.new(
        "RGB", (size[0]*cell_size, size[1]*cell_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for index, cell_value in np.ndenumerate(self.maze):
        x1 = index[1] * cell_size
        y1 = index[0] * cell_size
        x2 = (index[1] + 1) * cell_size
        y2 = (index[0] + 1) * cell_size

        if index == self.start:
            draw.rectangle((x1, y1, x2, y2), fill=(0, 255, 0))
        elif index == self.end:
            draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0))
        elif int(cell_value) in (0, 1):
            draw.rectangle((x1, y1, x2, y2), fill=wall_color)
        elif index in path:
            draw.rectangle((x1, y1, x2, y2), fill=path_color)

    image.save(filename)
