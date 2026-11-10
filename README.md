# Roop Face Swap on Google Colab: Free One-Click Video & Image Face Swapper (T4 GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/69yujirohanma/roop-new/blob/main/workingthebest.ipynb)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Google%20Colab-orange)
![GPU](https://img.shields.io/badge/GPU-T4%20%2F%20CUDA-green)

**Roop 1.1.0 on Google Colab** is a ready-to-run notebook for face swapping in videos and images, with no local install and no GPU of your own. Upload a source face and a target video, click **Run all**, and download the swapped result. It runs on the free Colab T4 GPU using `insightface` and the `inswapper_128` ONNX model, and works from the command line (no Tk GUI or webcam needed).

**Repository:** [github.com/69yujirohanma/roop-new](https://github.com/69yujirohanma/roop-new)

Disclaimer

This project is a face-swapping tool for lawful, consensual use only.

Consent is required. Only use faces and footage you own or have explicit permission to use. Never use this tool to impersonate, deceive, harass, defame, blackmail, or create sexual or intimate content of anyone, and never use images of minors.
No content filter. The NSFW check in this code is disabled. Nothing screens what you upload or generate, so responsibility for the output rests entirely with you.
Follow the law. Deepfake, privacy, copyright, and likeness laws vary by country and can carry criminal and civil penalties. You are responsible for knowing and obeying them.
Disclose synthetic media. Label generated content as AI-altered when you share it.
Model licences apply. The insightface models, including inswapper_128, are provided for non-commercial research use. Check each model's licence before any other use.
No warranty, no liability. This software is provided "as is", without warranty of any kind. The maintainers are not responsible for misuse, damages, or legal consequences arising from its use.
No affiliation. This repository is not affiliated with or endorsed by the original Roop, insightface, or GFPGAN projects or their authors.

If you cannot comply with these terms, do not use this software.

## Table of contents

- [Features](#features)
- [Quick start](#quick-start)
- [Options](#options)
- [Notebook layout](#notebook-layout)
- [How it works](#how-it-works)
- [Command-line reference](#command-line-reference)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Features

- **One-click Colab notebook:** open it, upload two files, press **Run all**.
- **Video and image face swap:** swap a face in an MP4 video or a single photo.
- **GPU accelerated:** uses CUDA through `onnxruntime-gpu` on the free T4, with CPU as a fallback.
- **Multiple faces:** optionally swap every face in the frame.
- **Optional face enhancer:** GFPGAN restoration for sharper results.
- **Keeps audio and frame rate:** the original soundtrack and fps are restored on the output video.
- **Colab-friendly fixes:**
  - Temporary frames are stored on local disk (`/tmp`) instead of next to the target, which is much faster if your video is on Google Drive.
  - `--keep-audio` and `--keep-fps` can actually be switched off (`--no-keep-audio`, `--no-keep-fps`).
  - Crash fixes for unreadable frames, very short videos, a missing `-o` argument, and output validation.

## Quick start

1. Open the notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/69yujirohanma/roop-new/blob/main/workingthebest.ipynb)
2. **Runtime → Change runtime type → T4 GPU.**
3. Open the **Files** panel and upload two files directly into `/content/` (not into `sample_data` or a subfolder):
   - `face1.jpg` - the source face image
   - `video1.mp4` - the target video
4. **Runtime → Run all.**
5. When it finishes, click the refresh icon in the Files panel, then right-click `/content/output.mp4` → **Download**.

If Colab asks for a restart after the install cell, restart the session and continue from the **Create the project files** cell.

You can also clone the repository and upload the notebook yourself:

```bash
git clone https://github.com/69yujirohanma/roop-new.git
```

## Options

The last cell (**Run**) has a form at the top:

| Option | Default | What it does |
| --- | --- | --- |
| `MANY_FACES` | off | Swap every face in the frame instead of only the left-most one |
| `ENHANCE` | off | Run the GFPGAN face enhancer after swapping (needs the optional GFPGAN cell, slower) |
| `KEEP_AUDIO` | on | Copy the original audio track into the output |
| `VIDEO_QUALITY` | 18 | Output quality, `0` = best, `51` = worst |

### Using different file names

Edit the command in the last cell and change the three paths:

```
-s "/content/face1.jpg"  -t "/content/video1.mp4"  -o "/content/output.mp4"
```

`-s` is the source face, `-t` the target image or video, `-o` the output. For an image target use something like `-t "/content/photo.png" -o "/content/output.png"`.

## Notebook layout

| Section | Purpose |
| --- | --- |
| GPU check | Prints the attached GPU, or a hint to enable one |
| 1. Install dependencies | `ffmpeg`, `insightface==0.7.3`, `onnxruntime-gpu`, `onnx`, `opennsfw2`, `psutil`, `tqdm`, `protobuf`, `pillow`, `cython` |
| Optional: GFPGAN | Installs `gfpgan` and patches `basicsr` for newer `torchvision` (only needed for `ENHANCE`) |
| 2. Create the project files | Writes the Roop source tree into `/content/roop` |
| 3. Run | Options form plus the command that does the swap |

### Files written to `/content/roop`

```
run.py                          entry point
roop/
  core.py                       argument parsing, pipeline, ffmpeg steps
  face_analyser.py              insightface detection (buffalo_l)
  globals.py, metadata.py, typing.py
  capturer.py, utilities.py     video helpers, ffmpeg wrappers, downloads
  predicter.py                  content-filter hooks
  ui.py, ui.json                Tk GUI (unused on Colab)
  processors/frame/
    core.py                     threaded frame processing
    face_swapper.py             inswapper_128 face swap
    face_enhancer.py            GFPGAN enhancer
```

## How it works

1. The source image is analysed with insightface to get the face embedding.
2. The target video is split into PNG frames with ffmpeg.
3. Each frame is processed in worker threads: faces are detected, then swapped with `inswapper_128.onnx`. Optionally GFPGAN cleans up the result.
4. The frames are re-encoded (`libx264` by default) at the original frame rate and the original audio is restored.

Models are downloaded on first run into `/content/roop/roop/models` (`inswapper_128.onnx`, plus `GFPGANv1.4.pth` if you use the enhancer). insightface downloads `buffalo_l` on first use.

## Command-line reference

The notebook calls the CLI directly, so you can use it yourself:

```
python run.py -s face.jpg -t video.mp4 -o out.mp4 [options]
```

| Flag | Description |
| --- | --- |
| `--frame-processor` | `face_swapper` (default), `face_enhancer`, or both |
| `--many-faces` | Process every face |
| `--keep-fps` / `--no-keep-fps` | Keep the original fps (default on) |
| `--keep-audio` / `--no-keep-audio` | Keep the original audio (default on) |
| `--keep-frames` | Keep the temporary frames |
| `--video-encoder` | `libx264` (default), `libx265`, `libvpx-vp9` |
| `--video-quality` | `0`-`51`, default `18` |
| `--execution-provider` | `cpu`, `cuda`, ... (auto-selects CUDA when available) |
| `--execution-threads` | Worker threads (default 4 on CUDA) |
| `--max-memory` | RAM limit in GB, `0` = unlimited |

## Troubleshooting

- **No GPU / very slow:** check the first cell prints a GPU. Without one, onnxruntime falls back to CPU.
- **Colab stops the run:** long videos can hit session or idle limits. Try a shorter clip, lower the resolution, or keep the tab open and active.
- **"No face in source path detected":** use a clear, front-facing photo with one visible face.
- **`ENHANCE` fails to import `gfpgan`:** run the optional GFPGAN cell first.
- **No audio in the result:** the target may have no audio track. The notebook then falls back to the silent video.
- **Files not found:** make sure the uploads are in `/content/` and the names match the command.

## FAQ

**Can I face swap a video for free on Google Colab?**
Yes. This notebook runs on the free Colab T4 GPU, so you don't need your own graphics card.

**Does it work on images as well as videos?**
Yes. Point `-t` at a photo (for example `photo.png`) and set `-o` to an image file.

**Can it swap more than one face?**
Yes. Turn on `MANY_FACES` to swap every face in each frame.

**Why can't I use the GUI or webcam mode?**
The Tk GUI and local-webcam *Live* button can't run on Colab, so this notebook uses the command line.

**Where do I find the result?**
It is saved to `/content/output.mp4`. Refresh the Files panel and download it.

---

**Keywords:** roop, face swap, face swapper, deepfake, video face swap, image face swap, Google Colab, Colab notebook, T4 GPU, insightface, inswapper, onnxruntime, GFPGAN, ffmpeg
