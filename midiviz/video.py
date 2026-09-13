"""Writes the finished video by feeding frames to ffmpeg.

We do not encode H.264 ourselves -- ffmpeg is the standard tool for that. It is
started as a separate program with two inputs: the raw frames, handed over
through its standard input as we paint them, and the WAV file of synthesized
audio. It combines ("muxes") them into one MP4.

Sending frames as we go means the whole video never has to fit in memory: a
three-minute 1080p video would be about 45 GB uncompressed.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile

from PyQt6.QtGui import QImage

#: Shown to the user when ffmpeg cannot be found.
FFMPEG_HINT = (
    "ffmpeg was not found. Install it and make sure it is on PATH.\n"
    "  Windows:  winget install Gyan.FFmpeg\n"
    "  macOS:    brew install ffmpeg\n"
    "  Linux:    sudo apt install ffmpeg"
)

#: Checked when ffmpeg is not on PATH, since installers often skip that step.
_COMMON_DIRS = [
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin",
]

#: Encoder settings per quality preset, as (crf, preset).
#: CRF is the quality dial: lower means better looking and larger. The preset
#: trades encoding time for file size at the same quality.
_QUALITY = {
    "high": ("17", "medium"),
    "balanced": ("21", "medium"),
    "fast": ("24", "veryfast"),
}


class FFmpegMissing(RuntimeError):
    """Raised when ffmpeg is not installed, so the caller can explain how to get it."""


def find_ffmpeg() -> str:
    """Locate the ffmpeg executable.

    Returns:
        The full path to ffmpeg.

    Raises:
        FFmpegMissing: if it cannot be found anywhere.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found

    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for directory in _COMMON_DIRS:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate

    raise FFmpegMissing(FFMPEG_HINT)


def _hidden_process_flags() -> dict:
    """Extra ``subprocess`` arguments that stop a console window flashing up.

    Only Windows needs this; elsewhere it returns nothing.
    """
    if os.name != "nt":
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return {"startupinfo": startup, "creationflags": subprocess.CREATE_NO_WINDOW}


def image_bytes(image: QImage) -> bytes:
    """Extract an image's raw red/green/blue/alpha bytes, ready to send to ffmpeg.

    Qt pads each row of pixels out to a multiple of four bytes. At the sizes we
    use there is never any padding, but if there were, this strips it -- ffmpeg
    expects rows packed tightly together.
    """
    width, height = image.width(), image.height()
    raw = image.constBits().asstring(image.sizeInBytes())

    row_bytes = image.bytesPerLine()
    if row_bytes == width * 4:
        return raw
    return b"".join(raw[y * row_bytes:y * row_bytes + width * 4] for y in range(height))


def _build_command(ffmpeg: str, audio_wav: str, out_path: str, fps: int,
                   width: int, height: int, quality: str) -> list[str]:
    """Assemble the ffmpeg command line.

    Input 0 is the raw frames arriving on standard input (the lone ``-``), and
    input 1 is the audio file.
    """
    crf, preset = _QUALITY.get(quality, _QUALITY["high"])
    return [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        # Input 0: raw pixels. ffmpeg cannot guess their size or layout, so
        # every detail has to be spelled out.
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        # Input 1: the synthesized audio.
        "-i", audio_wav,
        "-map", "0:v:0", "-map", "1:a:0",
        # H.264 video. yuv420p is the colour layout every player understands.
        "-c:v", "libx264", "-preset", preset, "-crf", crf,
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        # Put the index at the front so the file can start playing while it
        # is still downloading.
        "-movflags", "+faststart",
        "-shortest",            # stop at whichever stream ends first
        out_path,
    ]


def _write_frames(process: subprocess.Popen, renderer, n_frames: int,
                  progress, cancel) -> bool:
    """Paint each frame and hand it to ffmpeg through its standard input.

    Returns:
        True if ``cancel`` asked to stop. False once every frame has been sent,
        or as soon as ffmpeg quits early -- its exit code then says what failed.
    """
    for frame in range(n_frames):
        if cancel is not None and cancel():
            return True
        pixels = image_bytes(renderer.render(frame))
        try:
            process.stdin.write(pixels)
        except OSError:
            return False         # the pipe is broken: ffmpeg has already exited
        if progress is not None and (frame % 5 == 0 or frame == n_frames - 1):
            progress(frame / max(1, n_frames))
    return False


def encode_video(renderer, audio_wav: str, out_path: str, fps: int, n_frames: int,
                 width: int, height: int, quality: str = "high",
                 progress=None, cancel=None) -> str:
    """Render every frame, pipe it to ffmpeg, and mux it with the audio.

    Args:
        renderer: object with a ``render(frame_number)`` method returning a QImage.
        audio_wav: path to the WAV file to use as the audio track.
        out_path: where to write the MP4.
        fps: frames per second.
        n_frames: how many frames to render.
        width: frame width in pixels; must match what the renderer paints.
        height: frame height in pixels; must match what the renderer paints.
        quality: one of ``"high"``, ``"balanced"`` or ``"fast"``.
        progress: optional callback taking a float from 0.0 to 1.0.
        cancel: optional callback returning True to stop early.

    Returns:
        ``out_path``.

    Raises:
        FFmpegMissing: if ffmpeg is not installed.
        KeyboardInterrupt: if ``cancel`` asked to stop; the partial file is deleted.
        RuntimeError: if ffmpeg reports an error.
    """
    command = _build_command(find_ffmpeg(), audio_wav, out_path,
                             fps, width, height, quality)

    # ffmpeg's messages go to a temporary file rather than a pipe. Reading a
    # pipe would mean draining it while we write frames, and if we did not, a
    # chatty ffmpeg could fill the pipe's buffer and deadlock both programs.
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=errors,
                                       **_hidden_process_flags())
        except OSError as exc:
            raise FFmpegMissing(f"{FFMPEG_HINT}\n\n({exc})") from exc

        try:
            cancelled = _write_frames(process, renderer, n_frames, progress, cancel)
        finally:
            # Closing ffmpeg's input tells it there are no more frames, so it can
            # finish writing the file. This must happen even if we are bailing out.
            with contextlib.suppress(OSError):      # it may have exited already
                process.stdin.close()
            exit_code = process.wait()

        if cancelled:
            with contextlib.suppress(OSError):
                os.remove(out_path)                 # a half-written video is no use
            raise KeyboardInterrupt("Rendering cancelled")

        if exit_code != 0 or not os.path.exists(out_path):
            errors.seek(0)
            message = errors.read().decode("utf-8", "replace").strip()
            tail = "\n".join(message.splitlines()[-12:])
            raise RuntimeError(f"ffmpeg failed (exit {exit_code}).\n{tail}")

    if progress is not None:
        progress(1.0)
    return out_path


def reveal_in_file_manager(path: str) -> None:
    """Open the folder containing ``path`` with the file selected.

    Best effort: silently does nothing if the platform will not cooperate.
    """
    path = os.path.abspath(path)
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except OSError:
        pass
