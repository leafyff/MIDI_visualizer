"""Where the program looks for files and where it puts the ones it makes.

Everything lives in one folder, ``data/``, next to the program. Opening a file
starts there, finished videos are saved there, and the demo generator writes
there too, so there is only ever one place to look.

Keeping this in its own module means the rest of the program never has to work
out a path for itself.
"""

from __future__ import annotations

import os

#: The project folder: the one holding main.py, one level above this package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The folder for input MIDI files and finished videos.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def ensure_data_dir() -> str:
    """Create the data folder if it does not exist yet, and return its path.

    Safe to call as often as you like -- making a folder that already exists
    does nothing.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def resolve_input(path: str) -> str:
    """Find an input file, falling back to the data folder.

    This lets someone type ``demo.mid`` from anywhere and still get
    ``data/demo.mid``, without having to spell out the folder.

    Args:
        path: the file name or path that was asked for.

    Returns:
        The path as given if that file exists; otherwise the matching name
        inside the data folder if *that* exists. If neither does, the original
        path comes back unchanged, so the caller can report a missing file
        using the name the user actually typed.
    """
    if os.path.isfile(path):
        return path
    in_data = os.path.join(DATA_DIR, os.path.basename(path))
    return in_data if os.path.isfile(in_data) else path


def default_output_path(midi_path: str, suffix: str = ".mp4") -> str:
    """Where to save the result of converting ``midi_path``.

    The video takes the MIDI file's name and lands in the data folder, so
    ``lullaby.mid`` becomes ``data/lullaby.mp4``.

    Args:
        midi_path: the MIDI file being converted.
        suffix: the extension to give the output.
    """
    stem = os.path.splitext(os.path.basename(midi_path))[0]
    return os.path.join(ensure_data_dir(), stem + suffix)
