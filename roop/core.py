#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow

import roop.globals
import roop.metadata
from roop.predicter import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    # fix: store_true + default=True could never be switched off; BooleanOptionalAction adds --no-keep-audio etc.
    program.add_argument('--keep-fps', help='keep original fps (default on)', dest='keep_fps', action=argparse.BooleanOptionalAction, default=True)
    program.add_argument('--keep-audio', help='keep original audio (default on)', dest='keep_audio', action=argparse.BooleanOptionalAction, default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB (0 = unlimited)', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, cuda, ...)', dest='execution_provider', default=suggest_default_provider(), choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(args.source_path, args.target_path, args.output_path)
    roop.globals.frame_processors = args.frame_processor
    roop.globals.headless = bool(args.source_path or args.target_path or args.output_path)
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_audio = args.keep_audio
    roop.globals.keep_frames = args.keep_frames
    roop.globals.many_faces = args.many_faces
    roop.globals.video_encoder = args.video_encoder
    roop.globals.video_quality = args.video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    available = onnxruntime.get_available_providers()
    decoded = [provider for provider, encoded in zip(available, encode_execution_providers(available))
               if any(execution_provider in encoded for execution_provider in execution_providers)]
    # always keep CPU as a fallback so unsupported ops don't crash the session
    if 'CPUExecutionProvider' not in decoded:
        decoded.append('CPUExecutionProvider')
    return decoded


def suggest_max_memory() -> int:
    # fix: RLIMIT_DATA of 16GB is pointless/harmful on Colab (12-50GB, CUDA maps lots of address space)
    return 4 if platform.system().lower() == 'darwin' else 0


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_default_provider() -> List[str]:
    # use the GPU automatically when onnxruntime-gpu is installed
    return ['cuda'] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['cpu']


def suggest_execution_threads() -> int:
    # fix: this used to read roop.globals.execution_providers, which is still empty at argparse time
    available = onnxruntime.get_available_providers()
    if 'DmlExecutionProvider' in available or 'ROCMExecutionProvider' in available:
        return 1
    if 'CUDAExecutionProvider' in available:
        return 4
    return min(8, os.cpu_count() or 1)


def limit_resources() -> None:
    # prevent tensorflow (only used for the NSFW check) from grabbing the GPU
    try:
        for gpu in tensorflow.config.experimental.list_physical_devices('GPU'):
            tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
                tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
            ])
    except RuntimeError:
        pass  # devices were already initialised
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3  # fix: darwin branch used 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        import torch  # lazy: no need to import torch up front
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        update_status('ffmpeg/ffprobe is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        from roop import ui  # lazy: tkinter/customtkinter are not available on Colab
        ui.update_status(message)


def reject_target() -> None:
    update_status('Target was rejected by the content filter.')
    destroy()


def start() -> None:
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # content filter calls removed/bypassed
    if is_image(roop.globals.target_path):
        predict_image(roop.globals.target_path)
    if is_video(roop.globals.target_path):
        predict_video(roop.globals.target_path)
    output_directory = os.path.dirname(os.path.abspath(roop.globals.output_path))
    os.makedirs(output_directory, exist_ok=True)
    # process image to image
    if is_image(roop.globals.target_path):
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
            release_resources()
        # fix: validated the target instead of the output
        if is_image(roop.globals.output_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    update_status('Creating temp resources...')
    create_temp(roop.globals.target_path)
    update_status('Extracting frames...')
    extract_frames(roop.globals.target_path)
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if not temp_frame_paths:
        update_status('Frame extraction failed.')
        clean_temp(roop.globals.target_path)
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
        frame_processor.post_process()
        release_resources()
    # handles fps
    if roop.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(roop.globals.target_path)
    # handle audio
    if roop.globals.keep_audio:
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    else:
        move_temp(roop.globals.target_path, roop.globals.output_path)
    # clean and validate
    clean_temp(roop.globals.target_path)
    # fix: validate the output, not the target
    if is_video(roop.globals.output_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if roop.globals.headless:
        start()
    else:
        try:
            from roop import ui
            window = ui.init(start, destroy)
        except Exception as error:  # ImportError / TclError (no display, e.g. Colab)
            print(f'GUI unavailable ({error!r}). Use the CLI: python run.py -s face.jpg -t target.mp4 -o out.mp4')
            return
        window.mainloop()
