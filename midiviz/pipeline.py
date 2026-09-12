"""Runs the whole MIDI-to-video conversion, start to finish.

Both the window and the command line call :func:`render_to_video`, so there is
only one version of the process to understand and to keep working.

The four steps::

    MIDI file  ->  parse  ->  Score  -+->  synthesize  ->  WAV  -+->  ffmpeg  ->  MP4
                                      |                          |
                                      +->  animation  ->  frames +
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass

from .midi_parser import Score, parse_midi
from .renderer import FrameRenderer, VisualData
from .synth import render_score, write_wav
from .video import encode_video

#: Frame sizes offered in the interface, as (width, height) in pixels.
RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
}

#: What share of the progress bar each step is worth, so that it advances at a
#: roughly even pace. Measured on a 38 second piece: reading and planning take
#: under a tenth of a second, synthesizing the audio 1.5-1.7 seconds, and
#: painting and encoding the frames 14 seconds at 720p through 36 seconds at
#: 1080p. Encoding dominates, and the more so the larger the frames, so the
#: audio's share is set between the two extremes it was measured at.
STAGE_WEIGHTS = {"analyse": 0.02, "audio": 0.08, "video": 0.88, "finish": 0.02}

#: The same, for when the audio has already been synthesized and is simply
#: reused, which leaves the encoding as very nearly the whole job.
STAGE_WEIGHTS_REUSED_AUDIO = {"analyse": 0.02, "audio": 0.01,
                              "video": 0.95, "finish": 0.02}


@dataclass
class RenderSettings:
    """The choices made in the interface before rendering."""

    resolution: str = "1080p"    # a key of RESOLUTIONS
    fps: int = 30                # frames per second
    quality: str = "high"        # "high", "balanced" or "fast"
    reverb: float = 0.22         # how much room echo to add, 0.0 to 1.0

    @property
    def size(self) -> tuple[int, int]:
        """The chosen frame size as ``(width, height)``."""
        return RESOLUTIONS.get(self.resolution, RESOLUTIONS["1080p"])


class Cancelled(Exception):
    """Raised when the user stops a render before it finishes."""


class _ProgressScaler:
    """Turns each step's own 0-to-1 progress into one overall percentage.

    Each step reports its progress from 0.0 to 1.0 without knowing about the
    others. This maps that onto the step's slice of the bar, using
    :data:`STAGE_WEIGHTS`, so the overall figure only ever moves forwards.
    """

    def __init__(self, callback, weights: dict[str, float] | None = None):
        """
        Args:
            callback: called as ``callback(fraction, label)``, or None.
            weights: how much of the bar each step owns; defaults to
                :data:`STAGE_WEIGHTS`.
        """
        self._callback = callback
        self._weights = weights or STAGE_WEIGHTS
        self._base = 0.0      # where this step's slice begins, 0.0 to 1.0
        self._span = 1.0      # how wide this step's slice is
        self._label = ""

    def start(self, stage: str, label: str) -> None:
        """Begin a step named in :data:`STAGE_WEIGHTS` and show ``label``."""
        # This step's slice starts after every step listed before it.
        base = 0.0
        for name, weight in self._weights.items():
            if name == stage:
                break
            base += weight

        self._base = base
        self._span = self._weights[stage]
        self._label = label
        self(0.0)

    def __call__(self, fraction: float) -> None:
        """Report progress within the current step, from 0.0 to 1.0."""
        if self._callback:
            capped = 0.0 if fraction < 0.0 else (1.0 if fraction > 1.0 else fraction)
            self._callback(self._base + self._span * capped, self._label)


def render_to_video(midi_path: str, out_path: str,
                    settings: RenderSettings | None = None,
                    score: Score | None = None,
                    progress=None, cancel=None,
                    keep_wav: str | None = None,
                    audio_wav: str | None = None) -> str:
    """Convert a MIDI file into a video with sound.

    Args:
        midi_path: the ``.mid`` file to read.
        out_path: where to write the ``.mp4``.
        settings: resolution, frame rate and quality; defaults if omitted.
        score: an already-parsed score, to avoid reading the file twice.
        progress: optional callback taking ``(fraction, label)``.
        cancel: optional callback returning True to stop early.
        keep_wav: if given, the audio is written here and kept; otherwise it
            goes to a temporary file that is deleted afterwards.
        audio_wav: audio already synthesized for this score, which the caller
            still owns. Given one, the synthesis step is skipped entirely and
            the file is left alone -- it is about a third of the work. The
            caller is responsible for it matching: same score, and the same
            ``reverb`` as in ``settings``. A path that no longer exists is
            ignored and the audio is synthesized as usual.

    Returns:
        ``out_path``.

    Raises:
        ValueError: if the MIDI file has no notes to show.
        Cancelled: if ``cancel`` asked to stop.
        FFmpegMissing: if ffmpeg is not installed.
    """
    settings = settings or RenderSettings()

    # Check the file is really there before planning around it: it lives in a
    # temporary folder and something else may have cleaned it up.
    reuse_audio = bool(audio_wav) and os.path.isfile(audio_wav)
    report = _ProgressScaler(
        progress, STAGE_WEIGHTS_REUSED_AUDIO if reuse_audio else STAGE_WEIGHTS)

    def stop_requested() -> None:
        """Raise Cancelled if the caller has asked us to stop."""
        if cancel is not None and cancel():
            raise Cancelled()

    # --- 1. Read the MIDI and plan the animation ---------------------------
    report.start("analyse", "Reading MIDI…")
    if score is None:
        score = parse_midi(midi_path)
    if score.note_count == 0:
        raise ValueError("This MIDI file contains no notes to visualise.")
    stop_requested()
    report(0.4)

    width, height = settings.size
    data = VisualData(score, fps=settings.fps)
    renderer = FrameRenderer(data, width, height)
    report(1.0)
    stop_requested()

    # --- 2. Synthesize the audio, unless we were given it ------------------
    if reuse_audio:
        report.start("audio", "Reusing the preview's sound…")
        wav_path = audio_wav
        if keep_wav is not None:
            # Someone asked for a copy of the audio as well.
            shutil.copyfile(audio_wav, keep_wav)
        report(1.0)
    else:
        report.start("audio", "Synthesising audio…")
        audio = render_score(score, progress=report, cancel=cancel,
                             reverb=settings.reverb)
        stop_requested()

        wav_path = keep_wav
        if wav_path is None:
            handle, wav_path = tempfile.mkstemp(suffix=".wav", prefix="midiviz_")
            os.close(handle)  # we only wanted a unique name; write_wav opens it
        write_wav(wav_path, audio)
    stop_requested()

    # --- 3. Paint the frames and let ffmpeg combine everything -------------
    # Write beside the target and move the result into place only once it is
    # complete. Encoding straight to out_path would empty an existing video
    # the moment ffmpeg opened it, so cancelling a re-render -- or any failure
    # part way through -- would destroy a perfectly good file.
    #
    # The temporary name keeps the .mp4 extension, because that is how ffmpeg
    # decides which container to write, and sits in the same folder so that
    # moving it into place is a rename rather than a copy.
    folder = os.path.dirname(os.path.abspath(out_path))
    handle, work_path = tempfile.mkstemp(dir=folder, prefix=".midiviz_", suffix=".mp4")
    os.close(handle)
    try:
        report.start("video", "Rendering frames…")
        encode_video(renderer, wav_path, work_path,
                     fps=settings.fps, n_frames=data.n_frames,
                     width=width, height=height, quality=settings.quality,
                     progress=report, cancel=cancel)
        # Same folder, so this is a rename: it either happens or it does not.
        os.replace(work_path, out_path)
    except KeyboardInterrupt as exc:
        # encode_video signals cancellation this way; restate it in our terms.
        raise Cancelled() from exc
    finally:
        # Only clean up audio we made ourselves. Reused audio belongs to the
        # caller, who is still using it for the preview.
        made_it_ourselves = not reuse_audio and keep_wav is None
        for leftover in ((wav_path if made_it_ourselves else None), work_path):
            if leftover and os.path.exists(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass

    report.start("finish", "Finishing…")
    report(1.0)
    return out_path
