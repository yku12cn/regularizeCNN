"""
    Handle multiple tasks with tqdm bar in parallel.
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from torch.hub import tqdm


def _worker(job, jobargs, opened, filled):
    # Take a slot
    _slot = opened.pop()
    filled[os.getpid()] = _slot
    # Run the program
    result = job(**jobargs, vid=_slot)
    # Release the slot
    opened.append(filled.pop(os.getpid()))
    opened.sort(reverse=True)
    return result


def runParallelTqdm(func, arglist, workers=1):
    """Handle multiple tasks with tqdm bar in parallel.
       The function to be run must include keyword argument "vid",
       which should be passed to tqdm's position.

    Args:
        func (callable): The function you want to run in parallel
                        example: func(**kwarg, vid)
        arglist (dict/list of dict): arguments for specified function.
                        should be a list of keyword dictionaries.

        workers (int, optional): The number of processes run in parallel
                        At least 1, won't exceed the number of cpu cores.

    Returns:
        [list]: returns of your function in the same order of the arglist
    """
    if not isinstance(arglist, list):
        arglist = [arglist]
    workers = min(max(workers, 1), os.cpu_count())

    slotManager = Manager()
    opened = slotManager.list(range(workers - 1, -1, -1))
    filled = slotManager.dict()

    pb = tqdm(total=len(arglist), desc="Overall", leave=True,
              position=workers, ascii=(os.name == "nt"),
              unit="task", mininterval=0.2)

    executor = ProcessPoolExecutor(max_workers=workers)
    tasks = [executor.submit(_worker, func, args, opened, filled)
             for args in arglist]

    for _ in as_completed(tasks):
        # Adjust Overall progress bar position
        if len(executor._pending_work_items) < workers:
            pb.clear()
            pb.pos = (-max(filled.values()) - 1) if filled else 0
        pb.refresh()
        pb.update(1)

    executor.shutdown(wait=True)
    pb.close()
    return [task.result() for task in tasks]
