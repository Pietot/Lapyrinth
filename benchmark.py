""" Benchmarking the time and memory complexity of the different algorithms
    used to generate the maze."""


from multiprocessing import Process, Queue

import csv
import os
import timeit
import psutil

from tqdm import tqdm

from maze import Maze


def kruskal_time(iteration: int, size: int, queue: Queue):
    time = round(timeit.timeit(lambda: Maze(size).kruskal(),
                               number=iteration, globals=globals()), 5)
    queue.put(("Kruskal", size, time))


def prim_time(iteration: int, size: int, queue: Queue):
    time = round(timeit.timeit(lambda: Maze(size).prim(),
                               number=iteration, globals=globals()), 5)
    queue.put(("Prim", size, time))


def depth_first_search_time(iteration: int, size: int, queue: Queue):
    time = round(timeit.timeit(lambda: Maze(size).depth_first_search(),
                               number=iteration, globals=globals()), 5)
    queue.put(("Depth First Search", size, time))


def hunt_and_kill_time(iteration: int, size: int, queue: Queue):
    time = round(timeit.timeit(lambda: Maze(size).hunt_and_kill(),
                               number=iteration, globals=globals()), 5)
    queue.put(("Hunt and Kill", size, time))


def memory_usage(pid):
    process = psutil.Process(pid)
    return process.memory_info().rss


def kruskal_memory(size: int, queue: Queue):
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.kruskal()
    mem = memory_usage(os.getpid()) - mem
    queue.put((size, mem))


def prim_memory(size: int, queue: Queue):
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.prim()
    mem = memory_usage(os.getpid()) - mem
    queue.put((size, mem))


def depth_first_search_memory(size: int, queue: Queue):
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.depth_first_search()
    mem = memory_usage(os.getpid()) - mem
    queue.put((size, mem))


def hunt_and_kill_memory(size: int, queue: Queue):
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.hunt_and_kill()
    mem = memory_usage(os.getpid()) - mem
    queue.put((size, mem))


def memory_complexity() -> None:
    max_size = 105
    memory = {}
    for size in tqdm(range(5, max_size + 1, 10)):
        queues = []
        processes = []
        for func in [kruskal_memory, prim_memory, depth_first_search_memory, hunt_and_kill_memory]:
            queue = Queue()
            queues.append(queue)
            p = Process(target=func, args=(size, queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for i, queue in enumerate(queues):
            algo = ['Kruskal', 'Prim', 'Depth First Search', 'Hunt and Kill'][i]
            size, mem = queue.get()
            if algo not in memory:
                memory[algo] = {}
            memory[algo][size] = mem/1000

    with open('memory_complexity.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Size', 'Kruskal', 'Prim', 'Depth First Search', 'Hunt and Kill'])

        # Write the data rows
        for size in memory['Kruskal'].keys():
            writer.writerow([size,
                            memory['Kruskal'][size],
                            memory['Prim'][size],
                            memory['Depth First Search'][size],
                            memory['Hunt and Kill'][size]])


def time_complexity() -> None:
    """_summary_
    """
    max_size = 105
    execution_time = {}
    for size in tqdm(range(5, max_size + 1, 10)):
        queues = []
        processes = []
        for func in [kruskal_time, prim_time, depth_first_search_time, hunt_and_kill_time]:
            queue = Queue()
            queues.append(queue)
            p = Process(target=func, args=(1, size, queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for queue in queues:
            algo, size, time = queue.get()
            if algo not in execution_time:
                execution_time[algo] = {}
            execution_time[algo][size] = str(time).replace('.', ',')

    with open('time_complexity.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Size', 'Kruskal', 'Prim', 'Depth First Search', 'Hunt and Kill'])

        # Write the data rows
        for size in execution_time['Kruskal'].keys():
            writer.writerow([size,
                            execution_time['Kruskal'][size],
                            execution_time['Prim'][size],
                            execution_time['Depth First Search'][size],
                            execution_time['Hunt and Kill'][size]])


if __name__ == "__main__":
    tc = Process(target=time_complexity)
    mc = Process(target=memory_complexity)
    tc.start()
    mc.start()
    tc.join()
    mc.join()
