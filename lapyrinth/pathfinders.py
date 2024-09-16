"""A program capable of solving mazes with different path-finding algorithms.

Raises:
    UnsolvableMaze: If the algorithm cannot solve the maze in the configuration given.
    ValueError: If the following_direction is not 'left' or 'right'.

Returns:
    list[tuple[int, int]]: The path from the start to the end of the maze.
"""


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

# v1.6 :
# Start : 01/06/2024 at 17h05 FR
# End : 01/06/2024 at 21h20 FR
# Changelogs : Added Breadth First Search pathfinder

# v1.7 :
# Start : N/A
# End : N/A
# Changelogs : Added Gready Best First Search pathfinder

# v1.8 :
# Start : 08/06/2024 at 17h00 FR
# End :  08/06/2024 at N/A FR
# Changelogs : Added A* pathfinder

# v1.9 :
# Start : 17/06/2024 at 19h30 FR
# End :  17/06/2024 at  21h50 FR
# Changelogs : Added Dijkstra's pathfinder

import colorsys
import heapq
import random as rdm
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from lapyrinth import Maze


class UnsolvableMaze(Exception):
    """Exception class for unsolvable Mazes.

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


def left_hand(maze_obj: Maze) -> list[tuple[int, int]]:
    """Solves the maze with the left hand rule.

    It start by knowing if the left cell relative to the current direction is a wall or not.\n
    If it's not a wall, it will turn left, move forward and update the direction.\n
    Else, it checks if the front cell is a wall or not.\n
    If it's not a wall, it will move forward.\n
    Else, it will turn right and update the direction.\n

    To save the path, it will save the direction of the cell in a dictionary.

    Args:
        maze_obj (Maze): The maze object.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    direction_to_left: dict[tuple[int, int], tuple[int, int]] = {
        (0, 1): (-1, 0),
        (-1, 0): (0, -1),
        (0, -1): (1, 0),
        (1, 0): (0, 1),
    }
    current_cell: tuple[int, int] = maze_obj.start
    cell_with_directions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    direction = next(iter(direction_to_left))

    while current_cell != maze_obj.end:
        left_cell_col = current_cell[1] + direction_to_left[direction][1]
        left_cell_row = current_cell[0] + direction_to_left[direction][0]

        if maze_obj.maze[left_cell_row, left_cell_col] > 1:
            direction = turn_left(direction)
            update_cell_directions(
                cell_with_directions,
                current_cell,
                direction,
                algorithm="Left Hand Rule",
                error_message="Pathfinder is stuck in a loop or end is not reachable.",
            )
            current_cell = (left_cell_row, left_cell_col)
            continue

        front_cell_row = current_cell[0] + direction[0]
        front_cell_col = current_cell[1] + direction[1]

        if maze_obj.maze[front_cell_row, front_cell_col] > 1:
            update_cell_directions(
                cell_with_directions,
                current_cell,
                direction,
                algorithm="Left Hand Rule",
                error_message="Pathfinder is stuck in a loop or end is not reachable.",
            )
            current_cell = (front_cell_row, front_cell_col)
        else:
            update_cell_directions(
                cell_with_directions,
                current_cell,
                direction,
                algorithm="Left Hand Rule",
                error_message="Pathfinder is stuck in a loop or end is not reachable.",
            )
            direction = turn_right(direction)

    maze_obj.pathfinder = "Left Hand Rule"
    return directions_to_path(maze_obj, cell_with_directions)


def right_hand(maze_obj: Maze) -> list[tuple[int, int]]:
    """Solves the maze with the right hand rule.

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
        (-1, 0): (0, 1),
    }
    current_cell: tuple[int, int] = maze_obj.start
    cell_with_directions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    direction = next(iter(direction_to_right))

    while current_cell != maze_obj.end:
        left_cell_col = current_cell[1] + direction_to_right[direction][1]
        left_cell_row = current_cell[0] + direction_to_right[direction][0]

        if maze_obj.maze[left_cell_row, left_cell_col] > 1:
            direction = turn_right(direction)
            update_cell_directions(
                cell_with_directions,
                current_cell,
                direction,
                algorithm="Right Hand Rule",
                error_message="Pathfinder is stuck in a loop or end is not reachable.",
            )
            current_cell = (left_cell_row, left_cell_col)
            continue

        front_cell_row = current_cell[0] + direction[0]
        front_cell_col = current_cell[1] + direction[1]

        if maze_obj.maze[front_cell_row, front_cell_col] > 1:
            update_cell_directions(
                cell_with_directions,
                current_cell,
                direction,
                algorithm="Right Hand Rule",
                error_message="Pathfinder is stuck in a loop or end is not reachable.",
            )
            current_cell = (front_cell_row, front_cell_col)
        else:
            update_cell_directions(
                cell_with_directions,
                current_cell,
                direction,
                algorithm="Right Hand Rule",
                error_message="Pathfinder is stuck in a loop or end is not reachable.",
            )
            direction = turn_left(direction)

    maze_obj.pathfinder = "Right Hand Rule"
    return directions_to_path(maze_obj, cell_with_directions)


def random_mouse(maze_obj: Maze) -> list[tuple[int, int]]:
    """Solves the maze with the random mouse pathfinder.

    It will randomly choose a direction to move to until it reaches the end of the maze.\n
    For performance reasons, it will not choose the opposite direction until it's forced.\n

    Args:
        maze_obj (Maze): The maze object.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    current_cell = maze_obj.start
    path: list[tuple[int, int]] = [current_cell]
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    banned_direction = None

    while current_cell != maze_obj.end:
        neighbors = Maze.get_neighbors(
            maze_obj.maze, current_cell, directions=directions, return_visited=True
        )

        if banned_direction:
            neighbors = [neighbor for neighbor in neighbors if neighbor[1] != banned_direction]
        if not neighbors:
            neighbors = Maze.get_neighbors(
                maze_obj.maze, current_cell, directions=directions, return_visited=True
            )

        next_cell, direction = rdm.choice(neighbors)
        banned_direction = (-direction[0], -direction[1])
        path = update_path(path, next_cell)
        current_cell = next_cell

    maze_obj.pathfinder = "Random Mouse"
    return path


def pledge(maze_obj: Maze, following_direction: str) -> list[tuple[int, int]]:
    """Solves the maze with the Pledge pathfinder.

    While the end is not reached, the pathfinder go straightforward.\n
    If it reaches a wall, it will walk along the left/right wall defined by following_direction\n
    For each turn, it will increase or decrease a counter (depending on the direction)\n
    If the counter reaches 0, it will go back to moving straightforward and the cycle restart.

    Args:
        maze_obj (Maze): The maze object.
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
                (1, 0): (0, 1),
            }
        case "right":
            direction_to_turn: dict[tuple[int, int], tuple[int, int]] = {
                (0, 1): (1, 0),
                (1, 0): (0, -1),
                (0, -1): (-1, 0),
                (-1, 0): (0, 1),
            }
        case _:
            raise ValueError("Invalid following_direction (must be 'left' or 'right')")

    current_cell: tuple[int, int] = maze_obj.start
    direction = next(iter(direction_to_turn))
    path: list[tuple[int, int]] = [current_cell]
    nb_rotations = 0

    while current_cell != maze_obj.end:
        front_neighbor = (
            current_cell[0] + direction[0],
            current_cell[1] + direction[1],
        )

        if nb_rotations == 0:
            if maze_obj.maze[front_neighbor] > 1:
                current_cell = (
                    current_cell[0] + direction[0],
                    current_cell[1] + direction[1],
                )
                path = update_path(path, current_cell)
            else:
                direction = direction_to_turn[direction_to_turn[direction_to_turn[direction]]]
                nb_rotations += 1
        else:
            while nb_rotations != 0 and current_cell != maze_obj.end:
                front_neighbor = (
                    current_cell[0] + direction[0],
                    current_cell[1] + direction[1],
                )
                side_to_follow = (
                    current_cell[0] + direction_to_turn[direction][0],
                    current_cell[1] + direction_to_turn[direction][1],
                )

                if maze_obj.maze[side_to_follow] > 1:
                    direction = direction_to_turn[direction]
                    nb_rotations -= 1
                    current_cell = (
                        current_cell[0] + direction[0],
                        current_cell[1] + direction[1],
                    )
                    path = update_path(path, current_cell)
                elif maze_obj.maze[front_neighbor] < 2:
                    # Turning to one side equals turning to the other side 3 times
                    direction = direction_to_turn[direction_to_turn[direction_to_turn[direction]]]
                    nb_rotations += 1
                else:
                    current_cell = (
                        current_cell[0] + direction[0],
                        current_cell[1] + direction[1],
                    )
                    path = update_path(path, current_cell)

        if current_cell == maze_obj.start and nb_rotations == 0:
            raise UnsolvableMaze("Pledge", "End is not reachable.")

    maze_obj.pathfinder = "Pledge"
    return path


def dead_end_filler(maze_obj: Maze) -> list[tuple[int, int]]:
    """Solves the maze with the Dead End Filler pathfinder.

    It start by converting all path cells to 2.\n
    Then if will recursively fill the dead ends (cells with only one neighbor)
    until there is no more dead ends.\n
    Finally, it will try get the path from the start to the end of the maze.\n
    If it finds multiple paths, in other words, a cell with two or more neighbor,
    it will raise an UnsolvableMaze exception.

    Args:
        maze_obj (Maze): The maze object.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """

    def get_path() -> list[tuple[int, int]]:
        current_cell: tuple[int, int] = maze_obj.start
        path: list[tuple[int, int]] = [maze_obj.start]

        while current_cell != maze_obj.end:
            neighbors = Maze.get_neighbors(
                maze_obj.maze, current_cell, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
            )
            if len(neighbors) > 1:
                raise UnsolvableMaze(
                    "Dead End Filler",
                    "Pathfinder found multiple paths but it can't select one.",
                )
            if len(neighbors) == 0:
                raise UnsolvableMaze("Dead End Filler", "End is not reachable.")
            maze_obj.maze[current_cell] = 2
            current_cell = neighbors[0][0]
            path.append(current_cell)

        return path

    maze_obj.maze[maze_obj.maze > 1] = 3
    stack = get_dead_ends(maze_obj)

    while stack:
        rows, columns = zip(*stack)
        maze_obj.maze[rows, columns] = 2
        stack = get_dead_ends(maze_obj)

    maze_obj.pathfinder = "Dead End Filler"
    return get_path()


def depth_first_search(maze_obj: Maze) -> list[tuple[int, int]]:
    """Solve the maze with the Depth First Search pathfinder.

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
        maze_obj (Maze): The maze object.

    Raises:
        UnsolvableMaze: If the algorithm cannot solve the maze due to the end not being reachable.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """

    def get_direction_priority() -> tuple[tuple[int, int], ...]:
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        # We reverse the list because we dfs will pop the last element
        directions.sort(
            key=lambda direction: abs(maze_obj.start[0] + direction[0] - maze_obj.end[0])
            + abs(maze_obj.start[1] + direction[1] - maze_obj.end[1]),
            reverse=True,
        )
        return tuple(directions)

    maze_obj.maze[maze_obj.maze > 1] = 3
    path: list[tuple[int, int]] = [maze_obj.start]
    stack: list[tuple[tuple[int, int], list[tuple[int, int]]]] = [
        (maze_obj.start, [maze_obj.start])
    ]

    directions = get_direction_priority()

    while stack:
        current_cell, path = stack.pop()
        maze_obj.maze[current_cell] = 2

        if current_cell == maze_obj.end:
            maze_obj.pathfinder = "Depth First Search"
            return path

        # Condition to optimize the search
        if current_cell == (maze_obj.end[0], maze_obj.end[1] - 1):
            path.append(maze_obj.end)
            maze_obj.pathfinder = "Depth First Search"
            return path

        neighbors = Maze.get_neighbors(maze_obj.maze, current_cell, directions=directions)

        for chosen_neighbor, _ in neighbors:
            if maze_obj.maze[chosen_neighbor] == 3:
                stack.append((chosen_neighbor, path + [chosen_neighbor]))

    raise UnsolvableMaze("Depth First Search", "End is not reachable.")


def breadth_first_search(maze_obj: Maze) -> list[tuple[int, int]]:
    """Solves the maze with the Breadth First Search pathfinder.

    It starts by converting all path cells to 3 (unvisited).\n
    Then it creates a queue with the start cell.\n
    It creates a dictionary to store the path from a cell to the cell it came from.\n
    While the queue is not empty, it pops the first cell and get it as the current cell.\n
    It marks the current cell as visited.\n
    If the current cell is the end, it ends the loop and return the path.\n
    Else, it gets the neighbors of the current cell.\n
    Adds the neighbors to the queue and updates the came_from dictionary.\n
    If the loop ends, it raises an UnsolvableMaze exception.\n

    Args:
        maze_obj (Maze): The maze object.

    Raises:
        UnsolvableMaze: If the algorithm cannot solve the maze due to the end not being reachable.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    maze_obj.maze[maze_obj.maze > 1] = 3
    queue: deque[tuple[int, int]] = deque([maze_obj.start])
    came_from: dict[tuple[int, int], tuple[int, int]] = {maze_obj.start: (0, 0)}

    while queue:
        current_cell = queue.popleft()
        maze_obj.maze[current_cell] = 2

        if current_cell == maze_obj.end:
            maze_obj.pathfinder = "Breadth First Search"
            return reconstruct_path(maze_obj, came_from)

        for neighbor, _ in Maze.get_neighbors(
            maze_obj.maze, current_cell, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
        ):
            if neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current_cell

    raise UnsolvableMaze("Breadth First Search", "End is not reachable.")


def greedy_best_first_search(maze_obj: Maze) -> list[tuple[int, int]]:
    """Solves the maze with the Best First Search pathfinder.

    It starts by converting all path cells to 3 (unvisited).\n
    Then it creates a set with the start cell.\n
    It creates a dictionary to store the cell and the cell it came from.\n
    It creates a heuristic function to calculate the distance between a cell and the end.\n
    While the set is not empty, it get the current cell with the lowest heuristic value.\n
    If the current cell is the end, it ends the loop and return the path.\n
    Else, it marks the current cell as visited, remove it from the set
    and get the neighbors of the current cell.\n
    Adds the neighbors to the set and updates the came_from dictionary.\n
    If the loop ends, it raises an UnsolvableMaze exception.\n

    Args:
        maze_obj (Maze): The maze object.

    Raises:
        UnsolvableMaze: If the algorithm cannot solve the maze due to the end not being reachable.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """

    def heuristic(cell: tuple[int, int]) -> int:
        return abs(cell[0] - maze_obj.end[0]) + abs(cell[1] - maze_obj.end[1])

    maze_obj.maze[maze_obj.maze > 1] = 3
    cell_to_explore: list[tuple[int, tuple[int, int]]] = []
    heapq.heappush(cell_to_explore, (0, maze_obj.start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {maze_obj.start: (0, 0)}

    while cell_to_explore:
        _, current_cell = heapq.heappop(cell_to_explore)

        if current_cell == maze_obj.end:
            maze_obj.pathfinder = "Greedy Best First Search"
            return reconstruct_path(maze_obj, came_from)

        maze_obj.maze[current_cell] = 2
        neighbors = Maze.get_neighbors(
            maze_obj.maze, current_cell, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
        )

        for neighbor, _ in neighbors:
            heapq.heappush(cell_to_explore, (heuristic(neighbor), neighbor))
            came_from[neighbor] = current_cell

    raise UnsolvableMaze("Greedy Best First Search", "End is not reachable.")


def dijkstra(maze_obj: Maze) -> list[tuple[int, int]]:
    """Solves the maze with the Best First Search pathfinder.

    It starts by converting all path cells to 3 (unvisited).\n
    Then, it creates a min-heap with the start cell.\n
    It creates 2 dictionaries, first to store the cell and the cell it came from.\n
    Second to store the g_score (cost from the start to the current cell).\n
    While the min-heap is not empty, it get the current cell with the lowest g_score
    If the current cell is the end, it ends the loop and return the path.\n
    Else, for each neighbor, it calculates the cost to reach the neighbor.\n
    If the neighbor is not in the g_score or the cost is lower than the previous one,
    it updates the g_score and adds the neighbor to the min-heap.\n
    If the loop ends, it raises an UnsolvableMaze exception.\n

    Args:
        maze_obj (Maze): The maze object.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """

    maze_obj.maze[maze_obj.maze > 1] = 3
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], int] = {}
    cell_to_explore: list[tuple[int, tuple[int, int]]] = []
    g_score[maze_obj.start] = 0
    heapq.heappush(cell_to_explore, (0, maze_obj.start))

    while cell_to_explore:
        current_distance, current_cell = heapq.heappop(cell_to_explore)

        if current_cell == maze_obj.end:
            return reconstruct_path(maze_obj, came_from)
        neighbors = Maze.get_neighbors(
            maze_obj.maze, current_cell, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
        )
        for neighbor, _ in neighbors:
            tentative_g_score = current_distance + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                came_from[neighbor] = current_cell
                heapq.heappush(cell_to_explore, (tentative_g_score, neighbor))

    raise UnsolvableMaze("Dijkstra", "End is not reachable.")


def a_star(maze_obj: Maze) -> list[tuple[int, int]]:
    """Solves the maze with the A* pathfinder.

    It starts by converting all path cells to 3 (unvisited).\n
    Then it creates a min-heap with the start cell.\n
    It creates 3 dictionaries, first to store the cell and the cell it came from.\n
    Second to store the g_score (cost from the start to the current cell).\n
    Third to store the f_score (g_score + heuristic).\n
    While the min-heap is not empty, it get the current cell with the lowest f_score
    and remove it from the min-heap.\n
    If the current cell is the end, it ends the loop and return the path.\n
    Else, for each neighbor, it calculates the cost to reach the neighbor.\n
    If the neighbor is not in the g_score or the cost is lower than the previous one,
    it updates the g_score, the f_score and adds the neighbor to the min-heap.\n
    If the loop ends, it raises an UnsolvableMaze exception.\n

    Args:
        maze_obj (Maze): The maze object.

    Raises:
        UnsolvableMaze: If the algorithm cannot solve the maze due to the end not being reachable.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """

    def heuristic(cell: tuple[int, int]) -> int:
        return abs(cell[0] - maze_obj.end[0]) + abs(cell[1] - maze_obj.end[1])

    maze_obj.maze[maze_obj.maze > 1] = 3
    cell_to_explore: list[tuple[int, tuple[int, int]]] = []
    heapq.heappush(cell_to_explore, (0, maze_obj.start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {maze_obj.start: (0, 0)}
    g_score = {maze_obj.start: 0}
    f_score = {maze_obj.start: heuristic(maze_obj.start)}
    cost = 1

    while cell_to_explore:
        _, current_cell = heapq.heappop(cell_to_explore)
        maze_obj.maze[current_cell] = 2

        if current_cell == maze_obj.end:
            maze_obj.pathfinder = "A star"
            return reconstruct_path(maze_obj, came_from)

        neighbors = Maze.get_neighbors(
            maze_obj.maze, current_cell, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
        )

        for neighbor, _ in neighbors:
            tentative_g_score = g_score[current_cell] + cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_cell
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                heapq.heappush(cell_to_explore, (f_score[neighbor], neighbor))

    raise UnsolvableMaze("A*", "End is not reachable.")


def turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    """Rotates a direction 90 degrees clockwise.

    Args:
        direction (tuple[int, int]): The direction to rotate.

    Returns:
        tuple[int, int]: The rotated direction.
    """
    row, column = direction
    return (column, -row)


def turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    """Rotates a direction 90 degrees counterclockwise.

    Args:
        direction (tuple[int, int]): The direction to rotate.

    Returns:
        tuple[int, int]: The rotated direction.
    """
    row, column = direction
    return (-column, row)


def update_path(path: list[tuple[int, int]], new_cell: tuple[int, int]) -> list[tuple[int, int]]:
    """Updates the path with the new cell.

    Args:
        path (list[tuple[int, int]]): The path to update.
        new_cell (tuple[int, int]): The new cell to add.

    Returns:
        list[tuple[int, int]]: The updated path.
    """
    try:
        index = path.index(new_cell)
        path = path[: index + 1]
    except ValueError:
        path.append(new_cell)

    return path


def update_cell_directions(
    cell_with_direction: dict[tuple[int, int], list[tuple[int, int]]],
    current_cell: tuple[int, int],
    direction: tuple[int, int],
    algorithm: str,
    error_message: str = "",
) -> None:
    """Updates the cell_with_direction dictionary with the current cell and direction.

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
    directions = cell_with_direction.setdefault(current_cell, [])
    if direction in directions:
        raise UnsolvableMaze(algorithm, error_message)
    directions.append(direction)


def get_dead_ends(maze_obj: Maze) -> list[tuple[int, int]]:
    """Get the dead ends of the maze. Meaning the cells with only one neighbor.


    Args:
        maze_obj (Maze): The maze object.

    Returns:
        list[tuple[int, int]]: A list on indexes of the dead ends.
    """
    dead_ends: list[tuple[int, int]] = []
    start_neighbors = Maze.get_neighbors(
        maze_obj.maze, maze_obj.start, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
    )
    end_neighbors = Maze.get_neighbors(
        maze_obj.maze, maze_obj.end, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
    )
    start_neighbors = [neighbor[0] for neighbor in start_neighbors]
    end_neighbors = [neighbor[0] for neighbor in end_neighbors]
    unvisited_cells = list(zip(*np.where(maze_obj.maze > 2)))
    for index in unvisited_cells:
        # Skip the start and the end because they are not dead ends.
        if index in (maze_obj.start, maze_obj.end):
            continue
        # We also skip the neighbors of the start and the end
        # because we can't reach the start and the end as a neighbor.
        if (
            index in start_neighbors
            and len(
                Maze.get_neighbors(
                    maze_obj.maze, maze_obj.start, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
                )
            )
            == 1
        ):
            continue
        if (
            index in end_neighbors
            and len(
                Maze.get_neighbors(
                    maze_obj.maze, maze_obj.end, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
                )
            )
            == 1
        ):
            continue

        neighbors = Maze.get_neighbors(
            maze_obj.maze, index, directions=((-1, 0), (0, 1), (1, 0), (0, -1))
        )

        if len(neighbors) == 1:
            dead_ends.append(index)
            unvisited_cells.remove(index)

        if not unvisited_cells:
            break

    return dead_ends


def directions_to_path(
    maze_obj: Maze, cell_with_directions: dict[tuple[int, int], list[tuple[int, int]]]
) -> list[tuple[int, int]]:
    """Convert the cell_with_directions dictionary to a path.

    Args:
        maze_obj (Maze): The maze object.
        cell_with_directions (dict[tuple[int, int], list[tuple[int, int]]]):
            The dictionary to convert.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    path: list[tuple[int, int]] = [maze_obj.start]
    current_cell = maze_obj.start

    while current_cell != maze_obj.end:
        current_cell = (
            current_cell[0] + cell_with_directions[current_cell][-1][0],
            current_cell[1] + cell_with_directions[current_cell][-1][1],
        )
        path.append(current_cell)

    return path


def reconstruct_path(
    maze_obj: Maze, came_from: dict[tuple[int, int], tuple[int, int]]
) -> list[tuple[int, int]]:
    """Reconstruct the path from the start to the end of the maze.

    Args:
        maze_obj (Maze): The maze object.
        came_from (dict[tuple[int, int], tuple[int, int]]): The dictionary to reconstruct the path.

    Returns:
        list[tuple[int, int]]: The path from the start to the end of the maze.
    """
    path: deque[tuple[int, int]] = deque([maze_obj.end])
    current_cell = maze_obj.end

    while current_cell != maze_obj.start:
        current_cell = came_from[current_cell]
        path.appendleft(current_cell)

    return list(path)


def generate_path(
    maze_obj: Maze,
    path: list[tuple[int, int]],
    filename: str | None = None,
) -> None:
    """Generate a maze image from a maze object.

    Args:
        maze_obj (Maze): The maze object.
        path (list[tuple[int, int]]): The path to draw.
        filename (str | None, optional): The filename. Defaults to None.
    """
    size = maze_obj.maze.shape
    cell_size = 50
    path_length = len(path)

    filename = (
        filename + ".png"
        if filename
        else f"Maze_{size[0]//2}x{size[1]//2}_{maze_obj.pathfinder}.png"
    )

    image = Image.new("RGB", (size[1] * cell_size, size[0] * cell_size), (255, 255, 255))

    draw = ImageDraw.Draw(image)

    def get_color(step: int, total_steps: int) -> tuple[int, int, int]:
        # Adjust hue to go from red (0) to yellow (1/6)
        hue = 1 / 6 * (step / total_steps)
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        return int(r * 255), int(g * 255), int(b * 255)

    def draw_path() -> None:
        for index, cell_value in np.ndenumerate(maze_obj.maze):
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


def print_path(maze_obj: Maze, path: list[tuple[int, int]]) -> None:
    """Print the path of the maze.

    Args:
        maze (Maze): The maze object.
        path (list[tuple[int, int]]): The path to print.
    """
    for index, value in np.ndenumerate(maze_obj.maze):
        if index in path:
            print("o ", end="")
        elif value < 2:
            print("# ", end="")
        else:
            print("  ", end="")

        if index[1] == maze_obj.maze.shape[1] - 1:
            print()
