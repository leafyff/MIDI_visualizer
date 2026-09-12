"""Turns a Score into the pictures that become the video's frames.

The work is split in two, because they have very different costs:

1. :class:`VisualData` works out, **once**, how tall every bar should be in
   every frame of the whole video. It is all numpy array maths, so doing it up
   front is cheap, and afterwards drawing any frame -- or scrubbing around in
   the preview -- is only a table lookup.

2. :class:`FrameRenderer` paints a single frame with Qt's ``QPainter``. The
   parts that never move (background, labels, empty bar troughs) are painted
   once into a base image and copied for each frame.

How the MIDI data drives the animation:

===========  ===============================================================
size         note velocity and the note's envelope set pitch-bar height;
             a track's summed energy sets its track-bar length; the glow
             around a bar grows with that bar's energy
colour       one colour per track; a bar played by several tracks at once
             blends theirs; the background tint follows the loudest tracks
speed        the rate bars fall and the beat pulse fades both follow the
             tempo, so a faster song animates more sharply; each note's
             attack and release follow its own length
===========  ===============================================================
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QColor, QFont, QImage, QLinearGradient, QPainter, QPen,
                         QRadialGradient)

from .midi_parser import Score


def format_time(seconds: float) -> str:
    """Format a number of seconds as ``m:ss``, e.g. ``2:07``."""
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Restrict ``value`` to the range ``low``..``high``."""
    return low if value < low else (high if value > high else value)


def _with_alpha(color: QColor, alpha: float) -> QColor:
    """Copy ``color`` with a new transparency (0 = invisible, 255 = solid)."""
    faded = QColor(color)
    faded.setAlpha(int(_clamp(alpha, 0, 255)))
    return faded


@dataclass
class Theme:
    """The colours and fonts of the visualisation. Change these to restyle it."""

    bg_top: QColor = field(default_factory=lambda: QColor(9, 11, 19))
    bg_bottom: QColor = field(default_factory=lambda: QColor(16, 18, 30))
    text: QColor = field(default_factory=lambda: QColor(236, 240, 250))
    text_dim: QColor = field(default_factory=lambda: QColor(255, 255, 255, 110))
    trough: QColor = field(default_factory=lambda: QColor(255, 255, 255, 20))
    # Tried in order; the first font installed on the machine wins.
    font_family: tuple = ("Segoe UI", "Inter", "Helvetica Neue", "DejaVu Sans", "Arial")


@dataclass
class Layout:
    """Where everything sits in the frame, for one output size.

    The design was drawn at 1920x1080. Every measurement is multiplied by
    ``scale``, so the same layout works at any resolution.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        """Compute every position and size from the frame dimensions."""
        w, h = self.width, self.height
        self.scale = s = h / 1080.0

        self.margin = 64 * s
        self.header_y = 60 * s

        # Font sizes, with a floor so tiny renders stay readable.
        self.title_size = max(11.0, 32 * s)
        self.meta_size = max(9.0, 22 * s)
        self.label_size = max(8.0, 17 * s)
        self.small_size = max(7.0, 14 * s)

        # The stage: where the pitch bars stand on their baseline.
        self.stage_x = self.margin
        self.stage_w = w - 2 * self.margin
        self.stage_top = h * 0.135
        self.baseline_y = h * 0.575
        self.stage_h = self.baseline_y - self.stage_top
        self.reflection_h = h * 0.085

        # The panel below it, holding one horizontal bar per track.
        self.panel_top = h * 0.715
        self.panel_h = h * 0.205

        self.progress_y = h - 62 * s
        self.progress_h = max(3.0, 6 * s)


def _shift_bars(array: np.ndarray, distance: int) -> np.ndarray:
    """Slide an array along the bar axis, padding the vacated edge with zeros.

    Positive ``distance`` moves towards higher pitches, negative towards lower.
    Works for both the 2-D energy array and the 3-D colour array.
    """
    shifted = np.zeros_like(array)
    if distance > 0:
        shifted[:, distance:] = array[:, :-distance]
    else:
        shifted[:, :distance] = array[:, -distance:]
    return shifted


class VisualData:
    """Every animation value, for every frame, worked out ahead of time.

    The important arrays, where ``T`` is the number of frames and ``B`` the
    number of pitch bars:

    * ``bars``      ``(T, B)``     how tall each bar is, 0.0-1.0
    * ``peaks``     ``(T, B)``     the slow-falling marker above each bar
    * ``bar_rgb``   ``(T, B, 3)``  each bar's colour
    * ``tracks``    ``(T, n)``     how full each track's bar is, 0.0-1.0
    * ``beat``      ``(T,)``       the beat flash, 1.0 on a beat then fading
    * ``global_energy`` ``(T,)``   overall loudness, driving the background glow
    """

    def __init__(self, score: Score, fps: int = 30, max_bars: int = 88):
        """
        Args:
            score: the parsed MIDI.
            fps: frames per second of the finished video.
            max_bars: the most bars to draw. 88 matches a piano keyboard.
        """
        self.score = score
        self.fps = fps
        self.duration = score.duration
        self.n_frames = max(1, int(math.ceil(score.duration * fps)))
        self.times = np.arange(self.n_frames, dtype=np.float64) / fps

        # Round the pitch range out to whole octaves so the C labels line up.
        low, high = score.pitch_bounds()
        self.pitch_lo = max(0, (low // 12) * 12)
        self.pitch_hi = min(127, ((high // 12) + 1) * 12 - 1)
        self.pitch_span = max(1, self.pitch_hi - self.pitch_lo + 1)

        # Normally one bar per semitone. Only a piece spanning more than
        # max_bars semitones gets squeezed, and then every note still lands on
        # a bar rather than falling off the edge of the stage.
        self.n_bars = min(self.pitch_span, max_bars)
        self.n_tracks = max(1, len(score.tracks))

        self.bars = np.zeros((self.n_frames, self.n_bars), dtype=np.float32)
        self.bar_rgb = np.zeros((self.n_frames, self.n_bars, 3), dtype=np.float32)
        self.tracks = np.zeros((self.n_frames, self.n_tracks), dtype=np.float32)
        self.track_rgb = np.array([t.color for t in score.tracks] or [(255, 255, 255)],
                                  dtype=np.float32)

        # Tempo decides how lively the animation is. Clamped so a file claiming
        # an absurd tempo cannot make the picture unwatchable.
        self.bpm = max(40.0, min(220.0, score.avg_bpm))

        self._accumulate_notes()
        self._spread_sideways()
        self._normalise()
        self._apply_falling()
        self._build_beat_curve()

    # -- mapping pitches onto bars ---------------------------------------

    def bar_pos(self, pitch: float) -> float:
        """Where a pitch sits on the stage, as a fractional bar position."""
        return (pitch - self.pitch_lo) * self.n_bars / self.pitch_span

    def bar_of(self, pitch: int) -> int:
        """Which bar a note lights up."""
        return int(_clamp(self.bar_pos(pitch), 0, self.n_bars - 1))

    def frame_at_time(self, seconds: float) -> int:
        """The frame number showing a given moment."""
        return int(_clamp(seconds * self.fps, 0, self.n_frames - 1))

    # -- step 1: turn notes into per-frame energy -------------------------

    def _accumulate_notes(self) -> None:
        """Add every note's envelope into the bar, colour and track arrays.

        Each note contributes a rise-hold-fade curve, exactly like the ADSR
        envelope the synthesizer uses, so the picture moves with the sound.
        """
        for track in self.score.tracks:
            colour = np.array(track.color, dtype=np.float32)
            for note in track.notes:
                bar = self.bar_of(note.pitch)
                duration = max(0.03, note.duration)

                # Longer notes fade out more slowly, within sensible limits.
                release = _clamp(0.10 + duration * 0.45, 0.12, 0.85)
                attack = _clamp(0.012 + duration * 0.05, 0.012, 0.10)
                if track.is_drum:
                    release, attack = 0.22, 0.008   # drums are pure hit and decay

                first = max(0, int(note.start * self.fps))
                last = min(self.n_frames,
                           int(math.ceil((note.end + release) * self.fps)) + 1)
                if last <= first:
                    continue

                # Seconds since the note began, for each frame it is visible.
                elapsed = self.times[first:last] - note.start
                envelope = self._note_envelope(elapsed, duration, attack, release,
                                               track.is_drum)

                # Louder notes make taller bars. The power curve exaggerates the
                # difference so dynamics are easy to see.
                strength = envelope * (note.velocity / 127.0) ** 1.25

                self.bars[first:last, bar] += strength
                self.bar_rgb[first:last, bar] += strength[:, None] * colour
                self.tracks[first:last, track.index] += strength

        # Turn the summed colours into an average, weighted by how much each
        # track contributed to that bar.
        self.bar_rgb /= np.maximum(self.bars[:, :, None], 1e-6)
        np.clip(self.bar_rgb, 0.0, 255.0, out=self.bar_rgb)

    @staticmethod
    def _note_envelope(elapsed: np.ndarray, duration: float, attack: float,
                       release: float, is_drum: bool) -> np.ndarray:
        """The rise-hold-fade curve of one note, sampled at each frame.

        Args:
            elapsed: seconds since the note started, one entry per frame.
            duration: how long the note is held.
            attack: seconds to reach full height.
            release: seconds to fade away after the note ends.
            is_drum: drums drop to a lower held level, so they read as hits.

        Returns:
            Values from 0.0 to 1.0, the same length as ``elapsed``.
        """
        envelope = np.empty(elapsed.size, dtype=np.float32)
        sustain = 0.30 if is_drum else 0.62
        decay = max(0.18, duration * 0.9)

        rising = elapsed < attack
        envelope[rising] = np.maximum(elapsed[rising], 0.0) / attack

        holding = (~rising) & (elapsed <= duration)
        envelope[holding] = sustain + (1.0 - sustain) * np.exp(
            -(elapsed[holding] - attack) / decay)

        fading = elapsed > duration
        if fading.any():
            # Start the fade from wherever the hold phase had reached.
            level = sustain + (1.0 - sustain) * math.exp(
                -max(0.0, duration - attack) / decay)
            envelope[fading] = level * np.exp(
                -(elapsed[fading] - duration) / (release * 0.45))

        return np.clip(envelope, 0.0, 1.0)

    # -- step 2: let notes bleed into neighbouring bars --------------------

    def _spread_sideways(self) -> None:
        """Share a little of each note's energy with the bars either side.

        Without this, a chord shows up as a few lone spikes. Spreading turns
        them into rounded shapes that read as one gesture, the way a spectrum
        analyser does. The note's own bar always stays the tallest.
        """
        weights = ((1, 0.48), (2, 0.20))     # weight at one and two bars away

        spread = self.bars.copy()
        weighted_colour = self.bar_rgb * self.bars[:, :, None]

        for distance, weight in weights:
            for direction in (distance, -distance):
                neighbour = _shift_bars(self.bars, direction) * weight
                neighbour_colour = _shift_bars(self.bar_rgb, direction)
                spread += neighbour
                weighted_colour += neighbour_colour * neighbour[:, :, None]

        self.bars = spread
        self.bar_rgb = weighted_colour / np.maximum(spread[:, :, None], 1e-6)
        np.clip(self.bar_rgb, 0.0, 255.0, out=self.bar_rgb)

    # -- step 3: scale everything into the 0..1 the drawing code expects ----

    def _normalise(self) -> None:
        """Squash the raw sums into 0.0-1.0 without hard clipping.

        ``1 - exp(-x)`` rises quickly at first then flattens, so a single note
        is clearly visible while a dense chord approaches full height instead
        of overshooting and looking identical to every other loud moment.
        """
        self.bars = 1.0 - np.exp(-self.bars * 1.15)

        # Each track is measured against its own busy moments, so a quiet
        # instrument still fills its bar when it is playing its loudest.
        for i in range(self.tracks.shape[1]):
            column = self.tracks[:, i]
            active = column[column > 0.01]
            reference = np.percentile(active, 88) if active.size else 1.0
            self.tracks[:, i] = 1.0 - np.exp(-column / max(reference, 0.25) * 1.1)

        self.global_energy = self.bars.mean(axis=1)
        loud = np.percentile(self.global_energy, 92)
        self.global_energy = np.clip(self.global_energy / max(loud, 1e-3), 0.0, 1.4)

    # -- step 4: make the bars fall rather than snap -----------------------

    def _apply_falling(self) -> None:
        """Let bars drop gradually, and leave a slower marker at the peak.

        A bar jumps straight up when a note is struck but sinks at a steady
        rate, which is what gives the animation its bounce. Faster music uses a
        faster fall so the bars keep up with the notes.
        """
        step = 1.0 / self.fps
        bar_fall = (0.85 + self.bpm / 105.0) * step
        peak_fall = (0.45 + self.bpm / 240.0) * step
        track_fall = (0.70 + self.bpm / 130.0) * step

        self.peaks = np.zeros_like(self.bars)

        height = np.zeros(self.n_bars, dtype=np.float32)
        peak = np.zeros(self.n_bars, dtype=np.float32)
        colour = np.zeros((self.n_bars, 3), dtype=np.float32)
        track_height = np.zeros(self.tracks.shape[1], dtype=np.float32)

        for frame in range(self.n_frames):
            target = self.bars[frame]
            # Rise instantly to a new note, otherwise sink by one step.
            height = np.maximum(target, height - bar_fall)
            peak = np.maximum(height, peak - peak_fall)
            self.bars[frame] = height
            self.peaks[frame] = peak

            # Keep showing the last real colour while a bar is only sinking,
            # otherwise a decaying bar would fade to black.
            sounding = target > 1e-3
            colour[sounding] = self.bar_rgb[frame][sounding]
            self.bar_rgb[frame] = colour

            track_height = np.maximum(self.tracks[frame], track_height - track_fall)
            self.tracks[frame] = track_height

    # -- step 5: the beat flash --------------------------------------------

    def _build_beat_curve(self) -> None:
        """Work out the beat flash for every frame.

        Each beat starts a flash that fades away before the next one. The first
        beat of a bar flashes at full strength, the others more gently. Faster
        music gets a tighter flash so the pulses stay separate.
        """
        self.beat = np.zeros(self.n_frames, dtype=np.float32)
        beats = self.score.beats
        if not beats:
            return

        fade_rate = 6.0 + self.bpm / 14.0

        # Look up once, outside the per-frame loop, whether each beat starts a bar.
        downbeats = {round(t, 4) for t in self.score.downbeats}
        strengths = [1.0 if round(t, 4) in downbeats else 0.55 for t in beats]

        for frame in range(self.n_frames):
            now = self.times[frame]
            # The most recent beat at or before this frame.
            i = bisect.bisect_right(beats, now) - 1
            if i >= 0:
                self.beat[frame] = strengths[i] * math.exp(-(now - beats[i]) * fade_rate)

    # -- reading the results ------------------------------------------------

    def dominant_color(self, frame: int) -> QColor:
        """The blended colour of whichever tracks are loudest in this frame.

        Squaring each track's level before averaging lets the loudest track
        dominate rather than everything washing out to grey.
        """
        levels = self.tracks[frame]
        if levels.sum() < 1e-4:
            return QColor(90, 110, 190)      # a calm blue for silent moments
        weights = levels[:, None] ** 2
        rgb = ((self.track_rgb[:len(levels)] * weights).sum(axis=0)
               / max(weights.sum(), 1e-6))
        return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))


class FrameRenderer:
    """Paints frames of one :class:`VisualData` at a fixed resolution."""

    def __init__(self, data: VisualData, width: int = 1920, height: int = 1080,
                 theme: Theme | None = None):
        """
        Args:
            data: the pre-computed animation.
            width, height: the size of the frames to paint, in pixels.
            theme: colours and fonts; the default theme is used if omitted.
        """
        self.data = data
        self.score = data.score
        self.theme = theme or Theme()
        self.layout = Layout(width, height)

        # Filled in by _build_base(): the unchanging backdrop, and the empty
        # trough rectangle behind each track's bar.
        self._base: QImage | None = None
        self._track_troughs: list[QRectF] = []
        self._build_base()

    def _font(self, size: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
        """Build a font at the given point size, using the theme's family list."""
        font = QFont()
        font.setFamilies(list(self.theme.font_family))
        font.setPointSizeF(size)
        font.setWeight(weight)
        return font

    # -- the unchanging backdrop, painted once ------------------------------

    def _build_base(self) -> None:
        """Paint everything that is identical in every frame.

        Doing this once and copying the result per frame is far cheaper than
        repainting the gradients and text 30 times a second.
        """
        layout = self.layout
        image = QImage(layout.width, layout.height, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.black)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        self._paint_background(painter)
        self._paint_header(painter)
        self._paint_octave_ruler(painter)
        self._paint_track_rows(painter)

        # The empty groove the progress bar fills up.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 26))
        painter.drawRoundedRect(
            QRectF(layout.margin, layout.progress_y, layout.stage_w, layout.progress_h),
            layout.progress_h / 2, layout.progress_h / 2)

        painter.end()
        self._base = image

    def _paint_background(self, painter: QPainter) -> None:
        """Fill the frame with a vertical gradient and darken the edges."""
        layout, theme = self.layout, self.theme

        backdrop = QLinearGradient(0, 0, 0, layout.height)
        backdrop.setColorAt(0.0, theme.bg_top)
        backdrop.setColorAt(0.62, theme.bg_bottom)
        backdrop.setColorAt(1.0, theme.bg_top)
        painter.fillRect(0, 0, layout.width, layout.height, backdrop)

        # A vignette: darker towards the corners, which draws the eye inwards.
        vignette = QRadialGradient(QPointF(layout.width / 2, layout.baseline_y),
                                   layout.width * 0.75)
        vignette.setColorAt(0.0, QColor(0, 0, 0, 0))
        vignette.setColorAt(0.75, QColor(0, 0, 0, 40))
        vignette.setColorAt(1.0, QColor(0, 0, 0, 150))
        painter.fillRect(0, 0, layout.width, layout.height, vignette)

    def _paint_header(self, painter: QPainter) -> None:
        """Draw the file name on the left and the song's statistics on the right."""
        layout, theme = self.layout, self.theme

        painter.setPen(QPen(theme.text))
        painter.setFont(self._font(layout.title_size, QFont.Weight.DemiBold))
        title_box = QRectF(layout.margin, layout.header_y - layout.title_size * 1.4,
                           layout.stage_w * 0.6, layout.title_size * 2.0)
        # Shorten a long file name with an ellipsis rather than letting it run
        # into the statistics on the right.
        title = painter.fontMetrics().elidedText(
            self.score.name, Qt.TextElideMode.ElideRight, int(title_box.width()))
        painter.drawText(
            title_box,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), title)

        painter.setPen(QPen(theme.text_dim))
        painter.setFont(self._font(layout.meta_size))
        summary = (f"{len(self.score.tracks)} tracks   ·   "
                   f"{self.score.note_count} notes   ·   "
                   f"{self.score.avg_bpm:.0f} BPM")
        painter.drawText(
            QRectF(layout.width * 0.45, layout.header_y - layout.meta_size * 1.4,
                   layout.width * 0.55 - layout.margin, layout.meta_size * 2.0),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            summary)

    def _paint_octave_ruler(self, painter: QPainter) -> None:
        """Mark each C with a faint vertical line and a label under the baseline.

        These give the eye a fixed reference, so it is possible to tell which
        part of the keyboard a burst of bars is in.
        """
        layout = self.layout
        painter.setFont(self._font(layout.small_size))
        bar_width = layout.stage_w / self.data.n_bars

        for pitch in range(self.data.pitch_lo, self.data.pitch_hi + 1):
            if pitch % 12:                     # every C, and nothing else
                continue
            x = layout.stage_x + self.data.bar_pos(pitch) * bar_width

            # A line that fades out towards the top, so it never competes
            # with the bars themselves.
            fade = QLinearGradient(QPointF(0, layout.stage_top),
                                   QPointF(0, layout.baseline_y))
            fade.setColorAt(0.0, QColor(255, 255, 255, 0))
            fade.setColorAt(1.0, QColor(255, 255, 255, 22))
            painter.setPen(QPen(fade, max(1.0, layout.scale)))
            painter.drawLine(QPointF(x, layout.stage_top), QPointF(x, layout.baseline_y))

            painter.setPen(QPen(QColor(255, 255, 255, 26), max(1.0, layout.scale)))
            painter.drawLine(QPointF(x, layout.baseline_y + 4 * layout.scale),
                             QPointF(x, layout.baseline_y + 14 * layout.scale))

            # Octave numbering: MIDI note 60 is C4 in the common convention.
            painter.setPen(QPen(QColor(255, 255, 255, 80)))
            painter.drawText(
                QRectF(x - 20 * layout.scale, layout.baseline_y + 14 * layout.scale,
                       40 * layout.scale, 20 * layout.scale),
                int(Qt.AlignmentFlag.AlignCenter), f"C{pitch // 12 - 1}")

    def _paint_track_rows(self, painter: QPainter) -> None:
        """Draw each track's colour dot, name and empty bar trough.

        Records the trough rectangles in ``self._track_troughs`` so that
        :meth:`_draw_track_bars` knows where to draw the moving part.
        """
        layout = self.layout
        self._track_troughs = []

        track_count = len(self.score.tracks)
        if not track_count:
            return

        # Up to five tracks fit in one column; more are split across two.
        columns = 2 if track_count > 5 else 1
        rows = math.ceil(track_count / columns)
        gutter = 40 * layout.scale
        column_w = (layout.stage_w - (columns - 1) * gutter) / columns
        row_h = min(layout.panel_h / rows, 44 * layout.scale)
        name_w = column_w * 0.30
        dot = 9 * layout.scale

        for i, track in enumerate(self.score.tracks):
            column, row = divmod(i, rows)
            x = layout.stage_x + column * (column_w + gutter)
            middle_y = layout.panel_top + row * row_h + row_h / 2

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*track.color))
            painter.drawEllipse(QPointF(x + dot, middle_y), dot / 2, dot / 2)

            # The name, shortened with an ellipsis if it will not fit.
            painter.setPen(QPen(QColor(235, 240, 252, 225)))
            painter.setFont(self._font(layout.label_size))
            name_box = QRectF(x + dot * 2.2, middle_y - row_h / 2,
                              name_w - dot * 2.2, row_h)
            label = painter.fontMetrics().elidedText(
                track.name, Qt.TextElideMode.ElideRight, int(name_box.width()))
            painter.drawText(
                name_box,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), label)

            bar_h = max(6.0, 13 * layout.scale)
            trough = QRectF(x + name_w, middle_y - bar_h / 2, column_w - name_w, bar_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.theme.trough)
            painter.drawRoundedRect(trough, bar_h / 2, bar_h / 2)
            self._track_troughs.append(trough)

    # -- the moving parts, painted for each frame ---------------------------

    def render(self, frame: int) -> QImage:
        """Paint one frame and return it as an image.

        Args:
            frame: the frame number, counting from zero.
        """
        data, layout = self.data, self.layout
        frame = int(_clamp(frame, 0, data.n_frames - 1))

        # Qt copies the image lazily, so this is cheap until we paint on it.
        image = QImage(self._base)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        energy = float(data.global_energy[frame])
        beat = float(data.beat[frame])
        colour = data.dominant_color(frame)

        # A wash of colour behind the stage that swells with the music.
        glow = QRadialGradient(QPointF(layout.width / 2, layout.baseline_y),
                               layout.width * (0.34 + 0.16 * energy))
        glow.setColorAt(0.0, _with_alpha(colour, 14 + 46 * energy + 22 * beat))
        glow.setColorAt(1.0, _with_alpha(colour, 0))
        painter.fillRect(0, 0, layout.width, layout.height, glow)

        self._draw_pitch_bars(painter, frame)
        self._draw_baseline(painter, beat, colour)
        self._draw_track_bars(painter, frame)
        self._draw_progress(painter, frame, colour)

        painter.end()
        return image

    def _draw_pitch_bars(self, painter: QPainter, frame: int) -> None:
        """Draw the main row of bars: one per pitch, growing up from the baseline."""
        data, layout = self.data, self.layout
        heights = data.bars[frame]
        peaks = data.peaks[frame]
        colours = data.bar_rgb[frame]

        slot = layout.stage_w / data.n_bars
        gap = min(slot * 0.28, 7 * layout.scale)
        width = max(1.5, slot - gap)
        radius = min(width * 0.45, 7 * layout.scale)

        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(data.n_bars):
            level = float(heights[i])
            if level < 0.004:               # invisible, so skip the work
                continue

            height = level * layout.stage_h
            x = layout.stage_x + i * slot + gap / 2
            y = layout.baseline_y - height

            red, green, blue = colours[i]
            if red + green + blue < 12:      # a bar with no colour yet
                red, green, blue = 120, 140, 200
            colour = QColor(int(red), int(green), int(blue))

            self._draw_bar_glow(painter, x, y, width, height, radius, level, colour)

            # The bar itself, lit from the top so it looks rounded.
            body = QLinearGradient(QPointF(0, y), QPointF(0, layout.baseline_y))
            body.setColorAt(0.0, colour.lighter(int(130 + 45 * level)))
            body.setColorAt(0.55, colour)
            body.setColorAt(1.0, colour.darker(int(118 + 26 * (1.0 - level))))
            painter.setBrush(body)
            painter.drawRoundedRect(QRectF(x, y, width, height), radius, radius)

            # A bright cap on top, which makes the tip easy to follow.
            cap = min(height, max(2.0, 3.5 * layout.scale))
            painter.setBrush(QColor(255, 255, 255, int(90 + 120 * min(1.0, level))))
            painter.drawRoundedRect(QRectF(x, y, width, cap), cap / 2, cap / 2)

            self._draw_reflection(painter, x, width, height, radius, colour)

            # The peak marker, left behind as the bar sinks away from it.
            peak = float(peaks[i])
            if peak > level + 0.02:
                marker_h = max(1.5, 2.5 * layout.scale)
                painter.setBrush(_with_alpha(colour, 190))
                painter.drawRoundedRect(
                    QRectF(x, layout.baseline_y - peak * layout.stage_h,
                           width, marker_h),
                    marker_h / 2, marker_h / 2)

    def _draw_bar_glow(self, painter: QPainter, x: float, y: float, width: float,
                       height: float, radius: float, level: float,
                       colour: QColor) -> None:
        """Halo a bar with two translucent copies of itself, spreading with its energy.

        Two soft outlines are much cheaper than a real blur and look similar at
        this size.
        """
        if level <= 0.12:
            return
        for spread, alpha in ((9 * self.layout.scale, 26), (20 * self.layout.scale, 12)):
            faded = alpha * min(1.0, level * 1.4)
            if faded <= 1:
                continue
            painter.setBrush(_with_alpha(colour, faded))
            painter.drawRoundedRect(
                QRectF(x - spread / 2, y - spread * 0.7,
                       width + spread, height + spread * 0.7),
                radius + spread / 2, radius + spread / 2)

    def _draw_reflection(self, painter: QPainter, x: float, width: float,
                         height: float, radius: float, colour: QColor) -> None:
        """Mirror a bar below the baseline, fading out, as if on a polished floor."""
        layout = self.layout
        depth = min(height * 0.5, layout.reflection_h)
        if depth <= 2:
            return
        mirrored = QLinearGradient(QPointF(0, layout.baseline_y),
                                   QPointF(0, layout.baseline_y + depth))
        mirrored.setColorAt(0.0, _with_alpha(colour, 70))
        mirrored.setColorAt(1.0, _with_alpha(colour, 0))
        painter.setBrush(mirrored)
        painter.drawRoundedRect(QRectF(x, layout.baseline_y + 2, width, depth),
                                radius, radius)

    def _draw_baseline(self, painter: QPainter, beat: float, colour: QColor) -> None:
        """Draw the line the bars stand on, which brightens on every beat."""
        layout = self.layout
        painter.setPen(QPen(_with_alpha(colour, 40 + 150 * beat),
                            max(1.0, (1.2 + 2.2 * beat) * layout.scale)))
        painter.drawLine(QPointF(layout.stage_x, layout.baseline_y),
                         QPointF(layout.stage_x + layout.stage_w, layout.baseline_y))

    def _draw_track_bars(self, painter: QPainter, frame: int) -> None:
        """Fill each track's trough to show how active that instrument is."""
        layout = self.layout
        painter.setPen(Qt.PenStyle.NoPen)

        for i, (track, trough) in enumerate(zip(self.score.tracks, self._track_troughs)):
            level = float(self.data.tracks[frame, i])
            width = trough.width() * _clamp(level)
            height = trough.height()
            if width <= 1:
                continue
            colour = QColor(*track.color)

            # A soft halo once the track gets busy.
            if level > 0.25:
                pad = 5 * layout.scale
                painter.setBrush(_with_alpha(colour, 40 * min(1.0, level)))
                painter.drawRoundedRect(
                    QRectF(trough.x() - pad, trough.y() - pad,
                           width + 2 * pad, height + 2 * pad),
                    (height + 2 * pad) / 2, (height + 2 * pad) / 2)

            fill = QLinearGradient(QPointF(trough.x(), 0),
                                   QPointF(trough.x() + trough.width(), 0))
            fill.setColorAt(0.0, colour.darker(130))
            fill.setColorAt(1.0, colour.lighter(int(120 + 40 * level)))
            painter.setBrush(fill)
            painter.drawRoundedRect(QRectF(trough.x(), trough.y(), width, height),
                                    height / 2, height / 2)

            # A bright tip at the leading edge.
            cap = max(1.5, 2.5 * layout.scale)
            painter.setBrush(QColor(255, 255, 255, int(60 + 130 * level)))
            painter.drawRoundedRect(
                QRectF(trough.x() + width - cap, trough.y(), cap, height),
                cap / 2, cap / 2)

    def _draw_progress(self, painter: QPainter, frame: int, colour: QColor) -> None:
        """Draw the progress bar, its handle, and the elapsed / total time."""
        data, layout, theme = self.data, self.layout, self.theme
        now = frame / data.fps
        fraction = _clamp(now / max(data.duration, 1e-6))

        painter.setPen(Qt.PenStyle.NoPen)
        fill = QLinearGradient(QPointF(layout.margin, 0),
                               QPointF(layout.margin + layout.stage_w, 0))
        fill.setColorAt(0.0, _with_alpha(colour, 170))
        fill.setColorAt(1.0, QColor(255, 255, 255, 220))
        painter.setBrush(fill)
        painter.drawRoundedRect(
            QRectF(layout.margin, layout.progress_y,
                   layout.stage_w * fraction, layout.progress_h),
            layout.progress_h / 2, layout.progress_h / 2)

        if fraction > 0.001:
            painter.setBrush(QColor(255, 255, 255, 235))
            handle = layout.progress_h * 1.5
            painter.drawEllipse(
                QPointF(layout.margin + layout.stage_w * fraction,
                        layout.progress_y + layout.progress_h / 2),
                handle, handle)

        painter.setPen(QPen(theme.text_dim))
        painter.setFont(self._font(layout.small_size * 1.15))
        painter.drawText(
            QRectF(layout.margin, layout.progress_y + layout.progress_h + 8 * layout.scale,
                   layout.stage_w, 28 * layout.scale),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop),
            f"{format_time(now)} / {format_time(data.duration)}")
