import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import tempfile
import urllib.request  # fix: `import urllib` alone does not expose urllib.request
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

import roop.globals

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'roop'

# monkey patch ssl for mac
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-loglevel', roop.globals.log_level, *args]
    try:
        subprocess.run(commands, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as error:
        # fix: errors used to be swallowed silently
        print(error.stdout.decode(errors='ignore')[-800:])
    except FileNotFoundError:
        print('ffmpeg not found')
    return False


def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    try:
        output = subprocess.check_output(command).decode().strip().split('/')
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        return 30.0


def extract_frames(target_path: str) -> bool:
    temp_directory_path = get_temp_directory_path(target_path)
    return run_ffmpeg(['-i', target_path, '-pix_fmt', 'rgb24', os.path.join(temp_directory_path, '%04d.png')])


def create_video(target_path: str, fps: float = 30.0) -> bool:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    return run_ffmpeg(['-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.png'), '-c:v', roop.globals.video_encoder, '-crf', str(roop.globals.video_quality), '-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(['-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', output_path])
    if not done:  # e.g. target has no audio track
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return sorted(glob.glob(os.path.join(glob.escape(temp_directory_path), '*.png')))


def get_temp_directory_path(target_path: str) -> str:
    # efficiency: use local disk (/tmp) instead of the target's folder -
    # on Colab a target on Google Drive made every frame read/write extremely slow
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    return os.path.join(tempfile.gettempdir(), TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    return os.path.join(get_temp_directory_path(target_path), TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: Any) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        default_name = f'{source_name}-{target_name}{target_extension}'
        # fix: os.path.isdir(None) raised TypeError when -o was omitted
        if not output_path:
            return os.path.join(os.getcwd(), default_name)
        if os.path.isdir(output_path):
            return os.path.join(output_path, default_name)
    return output_path


def create_temp(target_path: str) -> None:
    Path(get_temp_directory_path(target_path)).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    return bool(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    os.makedirs(download_directory_path, exist_ok=True)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if os.path.exists(download_file_path):
            continue
        # fix: single request (was two), and a .part file so an interrupted
        # download is never mistaken for a finished model
        part_path = download_file_path + '.part'
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get('Content-Length', 0))
            with open(part_path, 'wb') as file, tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
                    progress.update(len(chunk))
        os.replace(part_path, download_file_path)


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
