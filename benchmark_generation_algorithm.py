""" Benchmarking the time and memory complexity of the different algorithms
    used to generate the maze."""


from multiprocessing import Process, Queue

import csv
import os
import timeit
import psutil

from tqdm import tqdm

from maze import Maze
from recursive_maze import RecursiveMaze

def kruskal_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).kruskal(),
                               number=1, globals=globals()), 5)
    queue.put(("Kruskal", size, time))


def prim_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).prim(),
                               number=1, globals=globals()), 5)
    queue.put(("Prim", size, time))


def randomized_depth_first_search_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).randomized_depth_first_search(),
                               number=1, globals=globals()), 5)
    queue.put(("Randomized Depth First Search", size, time))


def hunt_and_kill_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).hunt_and_kill(),
                               number=1, globals=globals()), 5)
    queue.put(("Hunt and Kill", size, time))


def eller_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).eller(),
                               number=1, globals=globals()), 5)
    queue.put(("Eller", size, time))


def recursive_division_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: RecursiveMaze(size).recursive_division(),
                               number=1, globals=globals()), 5)
    queue.put(("Recursive Division", size, time))


def binary_tree_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).binary_tree(),
                               number=1, globals=globals()), 5)
    queue.put(("Binary Tree", size, time))


def sidewinder_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).sidewinder(),
                               number=1, globals=globals()), 5)
    queue.put(("Sidewinder", size, time))


def growing_tree_new_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).growing_tree(mode='newest'),
                               number=1, globals=globals()), 5)
    queue.put(("Growing Tree (Newest)", size, time))


def growing_tree_mid_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).growing_tree(mode='middle'),
                               number=1, globals=globals()), 5)
    queue.put(("Growing Tree (Middle)", size, time))


def growing_tree_old_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).growing_tree(mode='oldest'),
                               number=1, globals=globals()), 5)
    queue.put(("Growing Tree (Oldest)", size, time))


def growing_tree_rand_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).growing_tree(mode='random'),
                               number=1, globals=globals()), 5)
    queue.put(("Growing Tree (Random)", size, time))


def growing_tree_mixed_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).growing_tree(mode='mixed'),
                               number=1, globals=globals()), 5)
    queue.put(("Growing Tree (Mixed)", size, time))


def aldous_broder_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).aldous_broder(),
                               number=1, globals=globals()), 5)
    queue.put(("Aldous Broder", size, time))


def wilson_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).wilson(),
                               number=1, globals=globals()), 5)
    queue.put(("Wilson", size, time))


def recursive_backtracking_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: RecursiveMaze(size).recursive_backtracking(),
                               number=1, globals=globals()), 5)
    queue.put(("Recursive Backtracking", size, time))


def recursive_kruskal_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: RecursiveMaze(size).recursive_kruskal(),
                               number=1, globals=globals()), 5)
    queue.put(("Recursive Kruskal", size, time))
    
def recursive_hunt_and_kill_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: RecursiveMaze(size).recursive_hunt_and_kill(),
                               number=1, globals=globals()), 5)
    queue.put(("Recursive Hunt and Kill", size, time))
    
def iterative_division_time(size: int, queue: Queue) -> None:
    time = round(timeit.timeit(lambda: Maze(size).iterative_division(),
                               number=1, globals=globals()), 5)
    queue.put(("Iterative Division", size, time))


def memory_usage(pid):
    process = psutil.Process(pid)
    return process.memory_info().rss


def kruskal_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.kruskal()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Kruskal", size, mem))


def prim_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.prim()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Prim", size, mem))


def randomized_depth_first_search_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.randomized_depth_first_search()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Randomized Depth First Search", size, mem))


def hunt_and_kill_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.hunt_and_kill()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Hunt and Kill", size, mem))


def eller_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.eller()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Eller", size, mem))


def recursive_division_memory(size: int, queue: Queue) -> None:
    maze = RecursiveMaze(size)
    mem = memory_usage(os.getpid())
    maze.recursive_division()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Recursive Division", size, mem))


def binary_tree_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.binary_tree()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Binary Tree", size, mem))


def sidewinder_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.sidewinder()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Sidewinder", size, mem))


def growing_tree_new_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.growing_tree(mode='newest')
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Growing Tree (Newest)", size, mem))


def growing_tree_mid_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.growing_tree(mode='middle')
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Growing Tree (Middle)", size, mem))


def growing_tree_old_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.growing_tree(mode='oldest')
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Growing Tree (Oldest)", size, mem))


def growing_tree_rand_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.growing_tree(mode='random')
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Growing Tree (Random)", size, mem))


def growing_tree_mixed_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.growing_tree(mode='mixed')
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Growing Tree (Mixed)", size, mem))


def aldous_broder_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.aldous_broder()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Aldous Broder", size, mem))


def wilson_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.wilson()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Wilson", size, mem))


def recursive_backtracking_memory(size: int, queue: Queue) -> None:
    maze = RecursiveMaze(size)
    mem = memory_usage(os.getpid())
    maze.recursive_backtracking()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Recursive Backtracking", size, mem))


def recursive_kruskal_memory(size: int, queue: Queue) -> None:
    maze = RecursiveMaze(size)
    mem = memory_usage(os.getpid())
    maze.recursive_kruskal()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Recursive Kruskal", size, mem))
    
def recursive_hunt_and_kill_memory(size: int, queue: Queue) -> None:
    maze = RecursiveMaze(size)
    mem = memory_usage(os.getpid())
    maze.recursive_hunt_and_kill()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Recursive Hunt and Kill", size, mem))
    
def iterative_division_memory(size: int, queue: Queue) -> None:
    maze = Maze(size)
    mem = memory_usage(os.getpid())
    maze.iterative_division()
    mem = memory_usage(os.getpid()) - mem
    queue.put(("Iterative Division", size, mem))


def time_complexity() -> None:
    """ Benchmarking the time complexity of the different algorithms used to generate the maze.
    """
    max_size = 10000
    execution_time = {}
    for size in tqdm(range(10, max_size + 1, 500)):
        queues = []
        processes = []
        for func in [kruskal_time, prim_time, randomized_depth_first_search_time, hunt_and_kill_time,
                     binary_tree_time, recursive_division_time, eller_time, sidewinder_time,
                     growing_tree_new_time, growing_tree_mid_time, growing_tree_old_time,
                     growing_tree_rand_time, growing_tree_mixed_time, aldous_broder_time,
                     wilson_time, recursive_backtracking_time, recursive_kruskal_time,
                     recursive_hunt_and_kill_time, iterative_division_time]:
            queue = Queue()
            queues.append(queue)
            p = Process(target=func, args=(size, queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for queue in queues:
            algo, size, time = queue.get()
            if algo not in execution_time:
                execution_time[algo] = {}
            execution_time[algo][size] = time

    with open('time_complexity.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Size', 'Kruskal', 'Prim', 'Randomized Depth First Search',
                        'Hunt and Kill', 'Binary Tree', 'Eller', 'Recursive Division', 'Sidewinder',
                         'Growing Tree (Newest)', 'Growing Tree (Middle)', 'Growing Tree (Oldest)',
                         'Growing Tree (Random)', 'Growing Tree (Mixed)', 'Aldous Broder',
                         'Wilson', 'Recursive Backtracking', 'Recursive Kruskal',
                         'Recursive Hunt and Kill', 'Iterative Division'])

        # Write the data rows
        for size in execution_time['Kruskal'].keys():
            writer.writerow([size,
                            execution_time['Kruskal'][size],
                            execution_time['Prim'][size],
                            execution_time['Randomized Depth First Search'][size],
                            execution_time['Hunt and Kill'][size],
                            execution_time['Binary Tree'][size],
                            execution_time['Eller'][size],
                            execution_time['Recursive Division'][size],
                            execution_time['Sidewinder'][size],
                            execution_time['Growing Tree (Newest)'][size],
                            execution_time['Growing Tree (Middle)'][size],
                            execution_time['Growing Tree (Oldest)'][size],
                            execution_time['Growing Tree (Random)'][size],
                            execution_time['Growing Tree (Mixed)'][size],
                            execution_time['Aldous Broder'][size],
                            execution_time['Wilson'][size],
                            execution_time['Recursive Backtracking'][size],
                            execution_time['Recursive Kruskal'][size],
                            execution_time['Recursive Hunt and Kill'][size],
                            execution_time['Iterative Division'][size]])


def memory_complexity() -> None:
    """ Benchmarking the memory complexity of the different algorithms used to generate the maze.
    """
    max_size = 10000
    memory = {}
    for size in tqdm(range(0, max_size + 1, 500)):
        if size == 0:
            size = 5
        queues = []
        processes = []
        for func in [kruskal_memory, prim_memory, randomized_depth_first_search_memory, hunt_and_kill_memory,
                     binary_tree_memory, eller_memory, recursive_division_memory, sidewinder_memory,
                     growing_tree_new_memory, growing_tree_mid_memory, growing_tree_old_memory,
                     growing_tree_rand_memory, growing_tree_mixed_memory, aldous_broder_memory,
                     wilson_memory, recursive_backtracking_memory, recursive_kruskal_memory,
                     recursive_hunt_and_kill_memory], iterative_division_memory:
            queue = Queue()
            queues.append(queue)
            p = Process(target=func, args=(size, queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for queue in queues:
            algo, size, mem = queue.get()
            if algo not in memory:
                memory[algo] = {}
            memory[algo][size] = mem/1000

    with open('memory_complexity.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Size', 'Kruskal', 'Prim', 'Randomized Depth First Search',
                        'Hunt and Kill', 'Binary Tree', 'Eller', 'Recursive Division', 'Sidewinder',
                         'Growing Tree (Newest)', 'Growing Tree (Middle)', 'Growing Tree (Oldest)',
                         'Growing Tree (Random)', 'Growing Tree (Mixed)', 'Aldous Broder',
                         'Wilson', 'Recursive Backtracking', 'Recursive Kruskal',
                         'Recursive Hunt and Kill', 'Iterative Division'])

        # Write the data rows
        for size in memory['Kruskal'].keys():
            writer.writerow([size,
                            memory['Kruskal'][size],
                            memory['Prim'][size],
                            memory['Randomized Depth First Search'][size],
                            memory['Hunt and Kill'][size],
                            memory['Binary Tree'][size],
                            memory['Eller'][size],
                            memory['Recursive Division'][size],
                            memory['Sidewinder'][size],
                            memory['Growing Tree (Newest)'][size],
                            memory['Growing Tree (Middle)'][size],
                            memory['Growing Tree (Oldest)'][size],
                            memory['Growing Tree (Random)'][size],
                            memory['Growing Tree (Mixed)'][size],
                            memory['Aldous Broder'][size],
                            memory['Wilson'][size],
                            memory['Recursive Backtracking'][size],
                            memory['Recursive Kruskal'][size],
                            memory['Recursive Hunt and Kill'][size],
                            memory['Iterative Division'][size]])


if __name__ == "__main__":
    tc = Process(target=time_complexity)
    mc = Process(target=memory_complexity)
    tc.start()
    mc.start()
    tc.join()
    mc.join()
