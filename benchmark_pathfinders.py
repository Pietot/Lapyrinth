"""Benchmarking the time and memory complexity of the different patfinders
used to solve a maze."""

from multiprocessing import Process, Queue

import csv
import os
import timeit
import psutil

from tqdm import tqdm

import pathfinders
from maze import Maze


def left_hand_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(lambda: pathfinders.left_hand(maze), number=1, globals=globals()),
        5,
    )
    queue.put(("Left Hand", size, time))


def right_hand_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(
            lambda: pathfinders.right_hand(maze), number=1, globals=globals()
        ),
        5,
    )
    queue.put(("Right Hand", size, time))


def random_mouse_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(
            lambda: pathfinders.random_mouse(maze), number=1, globals=globals()
        ),
        5,
    )
    queue.put(("Random Mouse", size, time))


def pledge_left_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(
            lambda: pathfinders.pledge(maze, "left"), number=1, globals=globals()
        ),
        5,
    )
    queue.put(("Pledge Left", size, time))


def pledge_right_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(
            lambda: pathfinders.pledge(maze, "right"), number=1, globals=globals()
        ),
        5,
    )
    queue.put(("Pledge Right", size, time))


def dead_end_filler_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(
            lambda: pathfinders.dead_end_filler(maze), number=1, globals=globals()
        ),
        5,
    )
    queue.put(("Dead End Filler", size, time))


def depth_first_search_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(
            lambda: pathfinders.depth_first_search(maze), number=1, globals=globals()
        ),
        5,
    )
    queue.put(("Depth First Search", size, time))


def breadth_first_search_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(
            lambda: pathfinders.breadth_first_search(maze), number=1, globals=globals()
        ),
        5,
    )
    queue.put(("Breadth First Search", size, time))


def greedy_best_first_search_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(
            lambda: pathfinders.greedy_best_first_search(maze),
            number=1,
            globals=globals(),
        ),
        5,
    )
    queue.put(("Greedy Best First Search", size, time))


def a_star_time(maze: Maze, size: int, queue: Queue) -> None:
    time = round(
        timeit.timeit(lambda: pathfinders.a_star(maze), number=1, globals=globals()), 5
    )
    queue.put(("A*", size, time))


def memory_usage(pid):
    process = psutil.Process(pid)
    return process.memory_info().rss


def left_hand_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.left_hand(maze)
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Left Hand", size, mem))


def right_hand_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.right_hand(maze)
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Right Hand", size, mem))


def random_mouse_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.random_mouse(maze)
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Random Mouse", size, mem))


def pledge_left_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.pledge(maze, "left")
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Pledge Left", size, mem))


def pledge_right_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.pledge(maze, "right")
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Pledge Right", size, mem))


def dead_end_filler_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.dead_end_filler(maze)
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Dead End Filler", size, mem))


def depth_first_search_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.depth_first_search(maze)
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Depth First Search", size, mem))


def breadth_first_search_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.breadth_first_search(maze)
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Breadth First Search", size, mem))


def greedy_best_first_search_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.greedy_best_first_search(maze)
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Greedy Best First Search", size, mem))


def a_star_memory(maze: Maze, size: int, queue: Queue) -> None:
    mem = memory_usage(os.getpid())
    pathfinders.a_star(maze)
    mem = memory_usage(os.getpid()) - mem
    queue.put(("A*", size, mem))


def time_complexity() -> None:
    """Benchmarking the time complexity of the different algorithms used to generate the maze."""
    max_size = 10000
    execution_time = {}
    for size in tqdm(range(max_size + 1, 500)):
        if size == 0:
            size = 5
        queues = []
        processes = []
        maze = Maze(size)
        maze.sidewinder()
        for func in [
            left_hand_time,
            right_hand_time,
            random_mouse_time,
            pledge_left_time,
            pledge_right_time,
            dead_end_filler_time,
            depth_first_search_time,
            breadth_first_search_time,
            greedy_best_first_search_time,
            a_star_time,
        ]:
            queue = Queue()
            queues.append(queue)
            process = Process(target=func, args=(maze, size, queue))
            processes.append(process)
            process.start()

        for p in processes:
            p.join()

        for queue in queues:
            algo, size, time = queue.get()
            if algo not in execution_time:
                execution_time[algo] = {}
            execution_time[algo][size] = time

    with open(
        "pathfinder_time_complexity.csv", "w", newline="", encoding="utf-8"
    ) as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(
            [
                "Size",
                "Left Hand",
                "Right Hand",
                "Random Mouse",
                "Pledge Left",
                "Pledge Right",
                "Dead End Filler",
                "Depth First Search",
                "Breadth First Search",
                "Greedy Best First Search",
                "A*",
            ]
        )

        for size in execution_time["Left Hand"].keys():
            writer.writerow(
                [
                    size,
                    execution_time["Left Hand"][size],
                    execution_time["Right Hand"][size],
                    execution_time["Random Mouse"][size],
                    execution_time["Pledge Left"][size],
                    execution_time["Pledge Right"][size],
                    execution_time["Dead End Filler"][size],
                    execution_time["Depth First Search"][size],
                    execution_time["Breadth First Search"][size],
                    execution_time["Greedy Best First Search"][size],
                    execution_time["A*"][size],
                ]
            )


def memory_complexity() -> None:
    """Benchmarking the memory complexity of the different algorithms used to generate the maze."""
    max_size = 10000
    memory = {}
    for size in tqdm(range(max_size + 1, 500)):
        if size == 0:
            size = 5
        queues = []
        processes = []
        maze = Maze(size)
        maze.sidewinder()
        for func in [
            left_hand_memory,
            right_hand_memory,
            random_mouse_memory,
            pledge_left_memory,
            pledge_right_memory,
            dead_end_filler_memory,
            depth_first_search_memory,
            breadth_first_search_memory,
            greedy_best_first_search_memory,
            a_star_memory,
        ]:
            queue = Queue()
            queues.append(queue)
            process = Process(target=func, args=(maze, size, queue))
            processes.append(process)
            process.start()

        for p in processes:
            p.join()

        for queue in queues:
            algo, size, mem = queue.get()
            if algo not in memory:
                memory[algo] = {}
            memory[algo][size] = mem / 1000

    with open(
        "pathfinder_memory_complexity.csv", "w", newline="", encoding="utf-8"
    ) as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(
            [
                "Size",
                "Left Hand",
                "Right Hand",
                "Random Mouse",
                "Pledge Left",
                "Pledge Right",
                "Dead End Filler",
                "Depth First Search",
                "Breadth First Search",
                "Greedy Best First Search",
                "A*",
            ]
        )

        # Write the data rows
        for size in memory["Left Hand"].keys():
            writer.writerow(
                [
                    size,
                    memory["Left Hand"][size],
                    memory["Right Hand"][size],
                    memory["Random Mouse"][size],
                    memory["Pledge Left"][size],
                    memory["Pledge Right"][size],
                    memory["Dead End Filler"][size],
                    memory["Depth First Search"][size],
                    memory["Breadth First Search"][size],
                    memory["Greedy Best First Search"][size],
                    memory["A*"][size],
                ]
            )


if __name__ == "__main__":
    tc = Process(target=time_complexity)
    mc = Process(target=memory_complexity)
    tc.start()
    mc.start()
    tc.join()
    mc.join()
