"""MIDI Visualizer - turn a MIDI file into a video.

Run it in one of three ways::

    python main.py                        open the app
    python main.py song.mid               open the app with a file loaded
    python main.py song.mid -o out.mp4    render without the interface

Naming an output file (with ``-o`` or ``--wav``) is what selects the command
line mode; otherwise the window opens. Run ``python main.py --help`` for the
full list of options.
"""

from __future__ import annotations

import argparse
import os
import sys

#: Characters that a plain Windows console cannot print, and what to use instead.
_ASCII_REPLACEMENTS = {"…": "...", "·": "-", "—": "-", "→": "->"}

#: Width of the command line progress bar, in characters.
_BAR_WIDTH = 46


def _make_safe_printer():
    """Build a function that makes text printable on this console.

    Windows consoles often use a legacy character set that cannot represent
    block characters or typographic punctuation, and printing one raises
    ``UnicodeEncodeError``. Rather than crash a render that is otherwise fine,
    text is swapped down to plain ASCII when necessary.

    Returns:
        A function taking a string and returning a printable version of it.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"

    def safe(text: str) -> str:
        """Return ``text`` if the console can print it, or a plain-ASCII version."""
        try:
            text.encode(encoding)
            return text                      # the console can print it as is
        except (UnicodeEncodeError, LookupError):
            pass
        for fancy, plain in _ASCII_REPLACEMENTS.items():
            text = text.replace(fancy, plain)
        try:
            text.encode(encoding)
            return text
        except (UnicodeEncodeError, LookupError):
            # Last resort: anything still unprintable becomes "?".
            return text.encode("ascii", "replace").decode("ascii")

    return safe


def _render_on_command_line(args) -> int:
    """Render a video without opening the window, showing a progress bar.

    Returns:
        The process exit code: 0 on success, non-zero on failure.
    """
    # Frames are painted with QPainter, which needs a Qt application object to
    # exist even though no window is ever shown. The reference must be kept:
    # if it is garbage-collected mid-render, Qt takes the process down with it.
    from PyQt6.QtGui import QGuiApplication
    qt_app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    assert qt_app is not None

    from midiviz.paths import default_output_path
    from midiviz.pipeline import RenderSettings, render_to_video
    from midiviz.video import FFmpegMissing

    out_path = args.output or default_output_path(args.midi)
    settings = RenderSettings(resolution=args.size, fps=args.fps, quality=args.quality)

    safe = _make_safe_printer()
    filled_char, empty_char = ("█", "░")
    if safe(filled_char + empty_char) != filled_char + empty_char:
        filled_char, empty_char = "#", "-"

    def show_progress(fraction: float, label: str) -> None:
        """Redraw the progress bar in place, using a carriage return."""
        filled = int(_BAR_WIDTH * fraction)
        bar = filled_char * filled + empty_char * (_BAR_WIDTH - filled)
        sys.stdout.write(safe(f"\r  {bar} {fraction * 100:5.1f}%  {label:<24}"))
        sys.stdout.flush()

    print(safe(f"\n  {os.path.basename(args.midi)}  ->  {out_path}"))
    try:
        render_to_video(args.midi, out_path, settings,
                        progress=show_progress, keep_wav=args.wav)
    except FFmpegMissing as exc:
        print(f"\n\n  {exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n\n  Cancelled.\n", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n\n  Failed: {type(exc).__name__}: {exc}\n", file=sys.stderr)
        return 1

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(safe(f"\n\n  Done  ·  {size_mb:.1f} MB  ·  {out_path}\n"))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Read the command line and either open the window or render a file.

    Args:
        argv: arguments to parse; the real command line is used if omitted.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="midi-visualizer",
        description="Turn a MIDI file into a video with animated bars.")
    parser.add_argument("midi", nargs="?", help="input .mid / .midi file")
    parser.add_argument("-o", "--output", help="output .mp4 path")
    parser.add_argument("--size", default="1080p", choices=["720p", "1080p", "1440p"])
    parser.add_argument("--fps", type=int, default=30, choices=[24, 30, 60])
    parser.add_argument("--quality", default="high",
                        choices=["high", "balanced", "fast"])
    parser.add_argument("--wav", help="also write the synthesized audio here")
    args = parser.parse_args(argv)

    if args.midi:
        # A bare name such as "demo.mid" is looked up in the data folder too.
        from midiviz.paths import DATA_DIR, resolve_input
        args.midi = resolve_input(args.midi)
        if not os.path.isfile(args.midi):
            parser.error(f"file not found: {args.midi} "
                         f"(also looked in {DATA_DIR})")

    # Asking for an output file means "just render it"; otherwise show the window.
    if args.midi and (args.output or args.wav):
        return _render_on_command_line(args)

    from midiviz.ui import run
    return run(args.midi)


if __name__ == "__main__":
    raise SystemExit(main())
