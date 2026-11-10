import os
import webbrowser
import tkinter
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps

import roop.globals
import roop.metadata
from roop.face_analyser import get_one_face
from roop.capturer import get_video_frame, get_video_frame_total
from roop.predicter import predict_frame
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

WEBCAM_RUNNING = False

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)
    return ROOT


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{roop.metadata.name} {roop.metadata.version}')
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.25)

    ctk.CTkButton(root, text='Select a face', cursor='hand2', command=lambda: select_source_path()).place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)
    ctk.CTkButton(root, text='Select a target', cursor='hand2', command=lambda: select_target_path()).place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)

    # fix: all switches now read the variable instead of toggling the global (keep_fps could drift out of sync)
    switches = [
        ('Keep fps', 'keep_fps', 0.1, 0.6),
        ('Keep frames', 'keep_frames', 0.1, 0.65),
        ('Keep audio', 'keep_audio', 0.6, 0.6),
        ('Many faces', 'many_faces', 0.6, 0.65),
    ]
    for text, name, relx, rely in switches:
        variable = ctk.BooleanVar(value=bool(getattr(roop.globals, name)))
        switch = ctk.CTkSwitch(root, text=text, variable=variable, cursor='hand2',
                               command=lambda n=name, v=variable: setattr(roop.globals, n, v.get()))
        switch.place(relx=relx, rely=rely)

    ctk.CTkButton(root, text='Start', cursor='hand2', command=lambda: select_output_path(start)).place(relx=0.15, rely=0.75, relwidth=0.2, relheight=0.05)
    ctk.CTkButton(root, text='Destroy', cursor='hand2', command=lambda: destroy()).place(relx=0.4, rely=0.75, relwidth=0.2, relheight=0.05)
    ctk.CTkButton(root, text='Preview', cursor='hand2', command=lambda: toggle_preview()).place(relx=0.65, rely=0.75, relwidth=0.2, relheight=0.05)
    ctk.CTkButton(root, text='Live', cursor='hand2', command=lambda: webcam_preview()).place(relx=0.40, rely=0.83, relwidth=0.2, relheight=0.05)

    status_label = ctk.CTkLabel(root, text=None, justify='center')
    status_label.place(relx=0.1, rely=0.9, relwidth=0.8)

    donate_label = ctk.CTkLabel(root, text='Become a GitHub Sponsor', justify='center', cursor='hand2')
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(text_color=ctk.ThemeManager.theme.get('RoopDonate').get('text_color'))
    donate_label.bind('<Button>', lambda event: webbrowser.open('https://github.com/sponsors/s0md3v'))

    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Preview')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))
    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(title='select an source image', initialdir=RECENT_DIRECTORY_SOURCE)
    if is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(source_path)
        source_label.configure(image=render_image_preview(source_path, (200, 200)))
    else:
        roop.globals.source_path = None
        source_label.configure(image=None)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(title='select an target image or video', initialdir=RECENT_DIRECTORY_TARGET)
    if is_image(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(target_path)
        target_label.configure(image=render_image_preview(target_path, (200, 200)))
    elif is_video(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(target_path)
        target_label.configure(image=render_video_preview(target_path, (200, 200)))
    else:
        roop.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT

    if is_image(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save image output file', defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save video output file', defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        roop.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(output_path)
        start()


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0):
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    capture.release()  # fix: was only released when no frame was read
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    return None


def toggle_preview() -> None:
    global WEBCAM_RUNNING

    if PREVIEW.state() == 'normal':
        WEBCAM_RUNNING = False  # closing the window also stops the live loop
        PREVIEW.withdraw()
    elif roop.globals.source_path and roop.globals.target_path:
        init_preview()
        update_preview()
        PREVIEW.deiconify()


def init_preview() -> None:
    if is_image(roop.globals.target_path):
        preview_slider.pack_forget()
    if is_video(roop.globals.target_path):
        preview_slider.configure(to=get_video_frame_total(roop.globals.target_path))
        preview_slider.pack(fill='x')
        preview_slider.set(0)


def show_frame(frame, max_size: Tuple[int, int]) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.contain(image, max_size, Image.LANCZOS)
    preview_label.configure(image=ctk.CTkImage(image, size=image.size))


def update_preview(frame_number: int = 0) -> None:
    if roop.globals.source_path and roop.globals.target_path:
        # fix: image targets were opened with VideoCapture and could yield None
        if is_video(roop.globals.target_path):
            temp_frame = get_video_frame(roop.globals.target_path, int(frame_number))
        else:
            temp_frame = cv2.imread(roop.globals.target_path)
        if temp_frame is None:
            return
        predict_frame(temp_frame)
        source_face = get_one_face(cv2.imread(roop.globals.source_path))  # once, not per processor
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            temp_frame = frame_processor.process_frame(source_face, temp_frame)
        show_frame(temp_frame, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT))


def webcam_preview() -> None:
    """Local webcam only - does not work on Colab (the webcam is on your PC, not the VM)."""
    global WEBCAM_RUNNING

    if not roop.globals.source_path:
        return
    source_face = get_one_face(cv2.imread(roop.globals.source_path))
    if source_face is None:
        update_status('No face in source path detected.')
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        update_status('No webcam found.')
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)

    preview_label.configure(image=None)
    preview_slider.pack_forget()
    PREVIEW.deiconify()
    frame_processors = get_frame_processors_modules(roop.globals.frame_processors)
    WEBCAM_RUNNING = True
    frame_count = 0
    try:
        # fix: loop had no exit condition; now stops when the preview window is closed.
        # fix: PREVIEW_MAX_* were local variables with width/height swapped
        while WEBCAM_RUNNING:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 60 == 0:
                predict_frame(frame)
            frame_count += 1
            for frame_processor in frame_processors:
                frame = frame_processor.process_frame(source_face, frame)
            show_frame(frame, (960, 540))
            try:
                ROOT.update()
            except tkinter.TclError:
                break
    finally:
        WEBCAM_RUNNING = False
        cap.release()
        try:
            PREVIEW.withdraw()
        except tkinter.TclError:
            pass
