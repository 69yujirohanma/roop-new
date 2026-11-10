import os
import math
import importlib
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import roop.globals

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check', 'pre_start', 'process_frame', 'process_frames',
    'process_image', 'process_video', 'post_process'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        module = importlib.import_module(f'roop.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(module, method_name):
                raise NotImplementedError(method_name)
    except (ImportError, NotImplementedError) as error:
        quit(f'Frame processor {frame_processor} crashed: {error!r}')
    return module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            FRAME_PROCESSORS_MODULES.append(load_frame_processor_module(frame_processor))
    return FRAME_PROCESSORS_MODULES


def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], update: Callable[[], None]) -> None:
    threads = max(1, roop.globals.execution_threads or 1)
    # fix: floor division gave 0 when frames < threads -> infinite loop
    chunk_size = max(1, math.ceil(len(temp_frame_paths) / threads))
    with ThreadPoolExecutor(max_workers=threads) as executor:
        queue = create_queue(temp_frame_paths)
        futures = []
        while not queue.empty():
            futures.append(executor.submit(process_frames, source_path, pick_queue(queue, chunk_size), update))
        for future in as_completed(futures):
            future.result()


def create_queue(temp_frame_paths: List[str]) -> 'Queue[str]':
    queue: 'Queue[str]' = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: 'Queue[str]', queue_per_future: int) -> List[str]:
    return [queue.get() for _ in range(queue_per_future) if not queue.empty()]


def process_video(source_path: str, frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(total=len(frame_paths), desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        multi_process_frame(source_path, frame_paths, process_frames, lambda: update_progress(progress))


def update_progress(progress: Any = None) -> None:
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    progress.set_postfix({
        'memory_usage': f'{memory_usage:05.2f}GB',
        'execution_providers': roop.globals.execution_providers,
        'execution_threads': roop.globals.execution_threads
    })
    progress.update(1)
