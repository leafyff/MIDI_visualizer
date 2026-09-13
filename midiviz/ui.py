"""The application window, built with PyQt6.

The design is deliberately plain: choose a file, look at the result. One window
shows three things in the same place, swapped by a ``QStackedWidget``:

* **empty**   -- a target to drop a MIDI file onto;
* **preview** -- the animation, which can be played or scrubbed through;
* **video**   -- the finished MP4, playing with its sound.

Rendering happens on a :class:`RenderWorker` thread. Qt draws the window from a
single thread, so any slow work must move off it or the window freezes. The
thread reports back with *signals*, which Qt delivers safely to the window.
"""

from __future__ import annotations

import math
import os
import threading
from typing import TypeVar

from PyQt6.QtCore import (QElapsedTimer, QPointF, QRectF, QSize, Qt, QThread,
                          QTimer, QUrl, pyqtSignal)
from PyQt6.QtGui import (QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap,
                         QPolygonF)
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame,
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QSizePolicy, QSlider,
                             QStackedWidget, QStyle, QStyleOptionSlider,
                             QVBoxLayout, QWidget)

from .midi_parser import parse_midi
from .paths import default_output_path, ensure_data_dir
from .pipeline import Cancelled, RESOLUTIONS, RenderSettings, render_to_video
from .renderer import FrameRenderer, VisualData, format_time
from .synth import render_score, write_wav
from .video import FFmpegMissing, reveal_in_file_manager

#: The preview is rendered at 720p whatever the export size is: big enough to
#: look right on screen, small enough to redraw smoothly while playing.
PREVIEW_W, PREVIEW_H = 1280, 720
PREVIEW_FPS = 30

#: Which page of the stacked widget each mode shows.
_PAGES = {"empty": 0, "preview": 1, "video": 2}

#: Where the playback volume starts, out of 100. This only affects listening
#: inside the app -- the audio written into a saved video is always at full
#: level, because it is synthesized separately.
DEFAULT_VOLUME = 80

#: Explains the distinction the volume slider makes, shown as its tooltip.
VOLUME_HINT = ("Playback volume in this window.\n"
               "The saved video file is not affected.")

#: Qt style sheet. The syntax is CSS-like: a widget type or ``#objectName``,
#: then the properties to apply. ``__CHEVRON__`` is replaced at runtime with
#: the path of the dropdown arrow drawn by :func:`_chevron_path`.
STYLE = """
QMainWindow, QWidget#root { background: #0c0e16; }
QLabel { color: #e8ecf6; }
QLabel#title { font-size: 20px; font-weight: 600; }
QLabel#subtitle, QLabel#status { color: #8a93ab; font-size: 12px; }
QLabel#fileinfo { color: #b9c2d6; font-size: 13px; }
QLabel#time { color: #8a93ab; font-size: 12px; }

QPushButton {
    background: #1b1f2e; color: #e8ecf6; border: 1px solid #2a3044;
    border-radius: 7px; padding: 8px 16px; font-size: 13px;
}
QPushButton:hover { background: #232839; border-color: #3a4258; }
QPushButton:pressed { background: #171b28; }
QPushButton:disabled { color: #555c70; background: #14171f; border-color: #20242f; }
QPushButton#primary {
    background: #4d7cff; border: none; color: #ffffff; font-weight: 600;
    padding: 9px 22px;
}
QPushButton#primary:hover { background: #5f8aff; }
QPushButton#primary:pressed { background: #3f6ae6; }
QPushButton#primary:disabled { background: #263354; color: #7d879e; }
QPushButton#ghost { background: transparent; border: 1px solid #2a3044; }
QPushButton#transport {
    background: #1b1f2e; border: 1px solid #2a3044; border-radius: 16px;
    min-width: 32px; max-width: 32px; min-height: 32px; max-height: 32px;
    padding: 0px; font-size: 13px;
}

QComboBox {
    background: #1b1f2e; color: #e8ecf6; border: 1px solid #2a3044;
    border-radius: 7px; padding: 7px 10px; font-size: 13px; min-width: 84px;
}
QComboBox:hover { border-color: #3a4258; }
QComboBox::drop-down {
    border: none; width: 20px; subcontrol-origin: padding;
    subcontrol-position: center right;
}
QComboBox::down-arrow { image: url(__CHEVRON__); width: 10px; height: 10px; }
QComboBox QAbstractItemView {
    background: #1b1f2e; color: #e8ecf6; border: 1px solid #2a3044;
    selection-background-color: #4d7cff; outline: none;
}

QSlider::groove:horizontal { height: 4px; background: #232839; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #4d7cff; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #ffffff; width: 12px; height: 12px;
    margin: -4px 0; border-radius: 6px;
}
QSlider::handle:horizontal:disabled { background: #3a4258; }

/* The volume slider is smaller and quieter looking than the timeline. */
QSlider#volume::groove:horizontal { height: 3px; background: #232839; }
QSlider#volume::sub-page:horizontal { background: #6b77a0; }
QSlider#volume::handle:horizontal {
    background: #b9c2d6; width: 10px; height: 10px;
    margin: -4px 0; border-radius: 5px;
}

QProgressBar {
    background: #171b28; border: none; border-radius: 4px;
    height: 8px; text-align: center; color: transparent;
}
QProgressBar::chunk { background: #4d7cff; border-radius: 4px; }

QFrame#stage { background: #06070c; border: 1px solid #1c2130; border-radius: 10px; }
"""

_WidgetT = TypeVar("_WidgetT", bound=QWidget)


def _named(widget: _WidgetT, name: str) -> _WidgetT:
    """Set a widget's object name: the ``#name`` that :data:`STYLE` refers to it by.

    PyQt also accepts ``objectName=`` as a constructor argument, but its type
    stubs do not declare it, so editors flag every such call as an error.

    Returns:
        ``widget`` itself, so it can be created and named in one expression.
    """
    widget.setObjectName(name)
    return widget


def _chevron_path() -> str:
    """Draw the little arrow for the dropdowns and save it to a temporary file.

    Qt style sheets can only *load* an arrow image, not draw one -- they have no
    way to rotate a shape into a chevron -- so we paint a tiny PNG instead.

    Returns:
        The file path, with forward slashes, as a style sheet needs.
    """
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "midiviz_chevron.png")
    image = QImage(20, 20, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(138, 147, 171), 2.2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline(QPolygonF([QPointF(5, 8), QPointF(10, 13), QPointF(15, 8)]))
    painter.end()

    image.save(path)
    return path.replace("\\", "/")


def _transport_icon(kind: str, size: int = 30) -> QIcon:
    """Draw the play triangle or the pause bars as an icon.

    The text characters that look like these symbols come out differently in
    every font -- the pause one in particular renders as two thick, widely
    spaced bars -- so the shapes are drawn here instead, at exactly the
    proportions we want.

    Args:
        kind: ``"play"`` or ``"pause"``.
        size: the icon's size in logical pixels.

    Returns:
        An icon ready for ``QPushButton.setIcon``.
    """
    # Draw at twice the size and tell Qt about it, so the edges stay sharp on
    # high resolution screens.
    scale = 2
    side = size * scale
    pixmap = QPixmap(side, side)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(232, 236, 246))

    if kind == "pause":
        bar_w = side * 0.105
        gap = side * 0.095
        bar_h = side * 0.38
        left = (side - (2 * bar_w + gap)) / 2
        top = (side - bar_h) / 2
        radius = bar_w * 0.4
        painter.drawRoundedRect(QRectF(left, top, bar_w, bar_h), radius, radius)
        painter.drawRoundedRect(QRectF(left + bar_w + gap, top, bar_w, bar_h),
                                radius, radius)
    else:
        # A triangle looks off-centre when centred by its bounding box, so it
        # is nudged right a little to sit properly inside the round button.
        width = side * 0.32
        height = side * 0.36
        x = (side - width) / 2 + side * 0.03
        y = (side - height) / 2
        painter.drawPolygon(QPolygonF([
            QPointF(x, y),
            QPointF(x + width, y + height / 2),
            QPointF(x, y + height),
        ]))
    painter.end()

    pixmap.setDevicePixelRatio(scale)
    return QIcon(pixmap)


def _speaker_icon(size: int = 18) -> QIcon:
    """Draw a small speaker, to label the volume slider."""
    scale = 2
    side = size * scale
    pixmap = QPixmap(side, side)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    colour = QColor(138, 147, 171)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colour)

    # The body: a small square with a cone opening out to the right.
    painter.drawPolygon(QPolygonF([
        QPointF(side * 0.16, side * 0.38), QPointF(side * 0.34, side * 0.38),
        QPointF(side * 0.55, side * 0.18), QPointF(side * 0.55, side * 0.82),
        QPointF(side * 0.34, side * 0.62), QPointF(side * 0.16, side * 0.62),
    ]))

    # Two arcs, suggesting sound coming out.
    pen = QPen(colour, side * 0.07)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for radius in (side * 0.16, side * 0.28):
        box = QRectF(side * 0.60 - radius / 2, side * 0.5 - radius,
                     radius, radius * 2)
        painter.drawArc(box, -60 * 16, 120 * 16)   # Qt measures in 1/16 degrees
    painter.end()

    pixmap.setDevicePixelRatio(scale)
    return QIcon(pixmap)


class SeekSlider(QSlider):
    """A position slider that jumps straight to wherever it is clicked.

    A plain ``QSlider`` treats a click on its groove as "step towards here",
    which in a media player is wrong: clicking the timeline should take you to
    that moment. It also only emits ``sliderMoved`` while the handle is being
    dragged, so a click would otherwise never reach the seeking code at all.

    This makes a click behave exactly like a drag to the same place: the handle
    goes there, ``sliderPressed`` / ``sliderMoved`` / ``sliderReleased`` all
    fire, and the player follows.
    """

    def mousePressEvent(self, event) -> None:
        """Treat a left click anywhere as grabbing the handle at that point."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)        # also emits sliderPressed
            self._seek_to(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Follow the pointer while the button is held."""
        if self.isSliderDown():
            self._seek_to(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Finish the drag, letting go of the handle."""
        if event.button() == Qt.MouseButton.LeftButton and self.isSliderDown():
            self._seek_to(event.position().x())
            self.setSliderDown(False)       # also emits sliderReleased
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _seek_to(self, x: float) -> None:
        """Move to the value at pixel position ``x`` and report it."""
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()

        # The handle has width, so the travel available to it is shorter than
        # the groove. Qt does this arithmetic for us once we know both.
        groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                      QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                      QStyle.SubControl.SC_SliderHandle, self)
        travel = groove.width() - handle.width()
        offset = int(x) - groove.x() - handle.width() // 2

        value = QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), offset, max(1, travel))
        self.setValue(value)
        self.sliderMoved.emit(value)


class PreviewCanvas(QWidget):
    """Shows one rendered frame, scaled to fit with black bars at the sides."""

    #: Emitted when the canvas is clicked, so clicking the picture plays it.
    clicked: pyqtSignal = pyqtSignal()

    def __init__(self, parent=None):
        """Create an empty canvas; call :meth:`set_image` to give it a frame."""
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image: QImage | None = None
        # The scaled copy actually drawn, kept so that redrawing an unchanged
        # frame (when the window regains focus, say) costs nothing.
        self._scaled: QPixmap | None = None
        self._scaled_size = (0, 0)

    def set_image(self, image: QImage | None) -> None:
        """Show a new frame and repaint."""
        self._image = image
        self._scaled = None
        self.update()          # asks Qt to call paintEvent soon

    def paintEvent(self, event) -> None:
        """Called by Qt whenever the widget needs redrawing."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(6, 7, 12))
        if self._image is None or self._image.isNull():
            painter.end()
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Scale to fit while keeping the shape, then centre what is left over.
        scale = min(self.width() / self._image.width(),
                    self.height() / self._image.height())
        target = (int(self._image.width() * scale), int(self._image.height() * scale))

        if self._scaled is None or self._scaled_size != target:
            self._scaled = QPixmap.fromImage(self._image.scaled(
                target[0], target[1], Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            self._scaled_size = target

        painter.drawPixmap((self.width() - target[0]) // 2,
                           (self.height() - target[1]) // 2, self._scaled)
        painter.end()

    def mousePressEvent(self, event) -> None:
        """Treat a click anywhere on the picture as play/pause."""
        self.clicked.emit()


class RenderWorker(QThread):
    """Runs the conversion on its own thread so the window stays responsive.

    Results come back as signals rather than return values, because the work
    finishes long after ``start()`` returns.
    """

    #: Progress so far (0.0-1.0) and the name of the current step.
    progressed: pyqtSignal = pyqtSignal(float, str)
    #: The finished video's path.
    finished_ok: pyqtSignal = pyqtSignal(str)
    #: A problem, as ``(kind, message)`` where kind is
    #: ``"cancelled"``, ``"ffmpeg"`` or ``"error"``.
    failed: pyqtSignal = pyqtSignal(str, str)

    def __init__(self, midi_path: str, out_path: str,
                 settings: RenderSettings, audio_wav: str | None = None,
                 parent=None):
        """Prepare a render. Call ``start()`` to run it on a new thread.

        Args:
            midi_path: the MIDI file to convert.
            out_path: where the finished video should be written.
            settings: resolution, frame rate and quality.
            audio_wav: sound already synthesized for this file, to use instead
                of making it again. The file is read, never removed.
            parent: the owning Qt object, so the thread is cleaned up with it.
        """
        super().__init__(parent)
        self.midi_path = midi_path
        self.out_path = out_path
        self.settings = settings
        self.audio_wav = audio_wav
        # An Event is the safe way to signal between threads; the worker polls
        # it between frames.
        self._stop = threading.Event()

    def cancel(self) -> None:
        """Ask the render to stop at the next opportunity."""
        self._stop.set()

    def run(self) -> None:
        """The thread's body. Qt calls this after ``start()``; never call it directly."""
        try:
            render_to_video(
                self.midi_path, self.out_path, self.settings,
                progress=lambda fraction, label: self.progressed.emit(fraction, label),
                cancel=self._stop.is_set,
                audio_wav=self.audio_wav,
            )
        except Cancelled:
            self.failed.emit("cancelled", "")
        except FFmpegMissing as exc:
            self.failed.emit("ffmpeg", str(exc))
        except Exception as exc:
            # Any other failure is shown in a dialog rather than crashing.
            self.failed.emit("error", f"{type(exc).__name__}: {exc}")
        else:
            self.finished_ok.emit(self.out_path)


class AudioPreparer(QThread):
    """Synthesizes the preview's sound in the background.

    The picture can be drawn the moment a file is loaded, but the sound has to
    be synthesized first, which takes a few seconds. Doing it on its own thread
    lets the preview appear straight away and gain sound when it is ready.
    """

    #: Path of the finished WAV file.
    ready: pyqtSignal = pyqtSignal(str)
    #: Something went wrong; the preview stays silent.
    failed: pyqtSignal = pyqtSignal(str)

    def __init__(self, score, reverb: float, parent=None):
        """
        Args:
            score: the parsed MIDI to synthesize.
            reverb: how much room echo to add. It must match what a saved
                video would use, so that the same file can be handed to the
                renderer instead of synthesizing it a second time.
        """
        super().__init__(parent)
        self.score = score
        self.reverb = reverb
        self._stop = threading.Event()

    def cancel(self) -> None:
        """Abandon the work, for when another file is loaded."""
        self._stop.set()

    def run(self) -> None:
        """Synthesize to a temporary WAV file. Qt calls this after ``start()``."""
        import tempfile

        try:
            audio = render_score(self.score, cancel=self._stop.is_set,
                                 reverb=self.reverb)
            if self._stop.is_set():
                return
            handle, path = tempfile.mkstemp(suffix=".wav", prefix=PREVIEW_WAV_PREFIX)
            os.close(handle)
            write_wav(path, audio)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            if self._stop.is_set():
                _delete_quietly(path)
            else:
                self.ready.emit(path)


#: Prefix for the temporary WAV files holding the preview's sound.
PREVIEW_WAV_PREFIX = "midiviz_preview_"


def _sweep_stale_preview_audio() -> int:
    """Delete preview sound files left behind by a previous run.

    These are removed when the window closes, but a session that is killed --
    by the task manager, a crash, or a power cut -- leaves its file behind, and
    they are tens of megabytes each. Windows refuses to delete a file another
    program still has open, so a copy in use by a second running window is
    skipped rather than pulled out from under it.

    Returns:
        How many files were removed.
    """
    import glob
    import tempfile

    removed = 0
    pattern = os.path.join(tempfile.gettempdir(), PREVIEW_WAV_PREFIX + "*.wav")
    for path in glob.glob(pattern):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass          # still in use, or not ours to delete
    return removed


def _delete_quietly(path: str | None) -> None:
    """Remove a file, ignoring the case where it is missing or still in use."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


class MainWindow(QMainWindow):
    """The application window."""

    def __init__(self):
        """Build the window. Nothing is loaded until a file is opened."""
        super().__init__()
        self.setWindowTitle("MIDI Visualizer")
        self.resize(1120, 780)
        self.setMinimumSize(860, 620)
        self.setAcceptDrops(True)

        # What is currently loaded. All None until a file is opened.
        self.midi_path: str | None = None
        self.score = None
        self.data: VisualData | None = None
        self.renderer: FrameRenderer | None = None
        self.video_path: str | None = None
        self.worker: RenderWorker | None = None

        # The preview's sound: synthesized in the background after loading.
        self.audio_worker: AudioPreparer | None = None
        self.preview_wav: str | None = None
        # What that sound was made from, as (midi path, reverb). A saved video
        # can reuse the file only when both still match.
        self.preview_wav_source: tuple[str, float] | None = None
        # The same, for a synthesis still in progress.
        self._pending_wav_source: tuple[str, float] | None = None
        # Where to jump to once the preview's sound has finished loading.
        self._preview_seek_ms: int | None = None

        # When a setting is changed mid-playback the video is rebuilt. Two
        # separate steps, and so two separate pieces of state:
        #   _resume_after_render  survives the rebuild, as (milliseconds, playing)
        #   _resume_ms            armed only once the new file is being loaded
        # Keeping them apart matters: closing the old file emits the very
        # signals that apply a resume, which would otherwise consume it against
        # the file we are about to replace.
        self._resume_after_render: tuple[int, bool] | None = None
        self._resume_ms: int | None = None
        self._resume_playing = False

        # Preview playback state.
        self.mode = "empty"
        self._preview_time = 0.0      # where the preview is, in seconds
        self._playing = False
        self._resume_after_scrub = False
        self._clock = QElapsedTimer()  # measures real time between frames
        self._timer: QTimer = QTimer(self)
        self._timer.setInterval(1000 // PREVIEW_FPS)
        self._timer.timeout.connect(self._advance_preview)

        self._build_ui()
        self._on_volume_changed(DEFAULT_VOLUME)
        self._set_mode("empty")

    # ------------------------------------------------------------------
    # Building the window
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create every widget and arrange it. Called once, from ``__init__``."""
        root = _named(QWidget(), "root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        heading = QVBoxLayout()
        heading.setSpacing(2)
        heading.addWidget(_named(QLabel("MIDI Visualizer"), "title"))
        heading.addWidget(_named(QLabel("Pick a MIDI file and turn it into a video."),
                                 "subtitle"))
        outer.addLayout(heading)

        outer.addWidget(self._build_stage(), 1)     # the 1 lets it take spare space
        outer.addLayout(self._build_transport())
        outer.addLayout(self._build_file_row())
        outer.addLayout(self._build_settings_row())
        outer.addLayout(self._build_progress_row())

        self.setStyleSheet(STYLE.replace("__CHEVRON__", _chevron_path()))

    def _build_stage(self) -> QFrame:
        """Build the big picture area: the drop target, preview and video player."""
        frame = _named(QFrame(), "stage")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(1, 1, 1, 1)

        # A stacked widget shows one of its children at a time.
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.stack.addWidget(self._build_empty_page())

        self.canvas = PreviewCanvas()
        self.canvas.clicked.connect(self._toggle_play)
        self.stack.addWidget(self.canvas)

        # The video player: a widget to show the picture, and an audio output
        # to play the sound. Both must be attached to the player.
        self.video = QVideoWidget()
        self.video.setStyleSheet("background: #06070c;")
        self.player: QMediaPlayer = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self._on_player_position)
        self.player.durationChanged.connect(self._on_player_duration)
        self.player.playbackStateChanged.connect(self._on_player_state)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.stack.addWidget(self.video)

        # A second player, for the preview's sound only. It has no video
        # output: the picture comes from the canvas, and this supplies the
        # sound and the clock that keeps the two together.
        self.preview_player: QMediaPlayer = QMediaPlayer(self)
        self.preview_audio_out = QAudioOutput(self)
        self.preview_player.setAudioOutput(self.preview_audio_out)
        self.preview_player.mediaStatusChanged.connect(self._on_preview_media_status)

        return frame

    @staticmethod
    def _build_empty_page() -> QWidget:
        """Build the 'drop a file here' placeholder."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(14)

        icon = QLabel("♪")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        big = QFont()
        big.setPointSize(44)
        icon.setFont(big)
        icon.setStyleSheet("color: #3a4258;")
        layout.addWidget(icon)

        headline = QLabel("Drop a MIDI file here")
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        headline.setStyleSheet("color: #b9c2d6; font-size: 15px;")
        layout.addWidget(headline)

        hint = _named(QLabel("or use Open MIDI file below   ·   .mid and .midi"),
                      "subtitle")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
        return page

    def _build_transport(self) -> QHBoxLayout:
        """Build the play button, position slider and time readout."""
        row = QHBoxLayout()
        row.setSpacing(10)

        self.play_icon = _transport_icon("play")
        self.pause_icon = _transport_icon("pause")
        self.play_btn: QPushButton = _named(QPushButton(), "transport")
        self.play_btn.setIcon(self.play_icon)
        self.play_btn.setIconSize(QSize(30, 30))
        self.play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self.play_btn)

        # The slider counts 0-1000 rather than seconds, so the same widget
        # works for any length of song.
        self.slider = SeekSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderPressed.connect(self._on_scrub_start)
        self.slider.sliderMoved.connect(self._on_scrub)
        self.slider.sliderReleased.connect(self._on_scrub_end)
        row.addWidget(self.slider, 1)

        self.time_label = _named(QLabel("0:00 / 0:00"), "time")
        self.time_label.setMinimumWidth(84)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight |
                                     Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.time_label)

        row.addSpacing(14)
        speaker = QLabel()
        speaker.setPixmap(_speaker_icon().pixmap(QSize(18, 18)))
        speaker.setToolTip(VOLUME_HINT)
        row.addWidget(speaker)

        self.volume_slider = _named(SeekSlider(Qt.Orientation.Horizontal), "volume")
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(DEFAULT_VOLUME)
        self.volume_slider.setFixedWidth(94)
        self.volume_slider.setToolTip(VOLUME_HINT)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        row.addWidget(self.volume_slider)

        # Only useful once a video exists, so it starts hidden.
        self.view_btn: QPushButton = _named(QPushButton("Back to preview"), "ghost")
        self.view_btn.clicked.connect(self._toggle_view)
        self.view_btn.hide()
        row.addWidget(self.view_btn)
        return row

    def _build_file_row(self) -> QHBoxLayout:
        """Build the Open button and the description of the loaded file."""
        row = QHBoxLayout()
        row.setSpacing(12)
        self.open_btn: QPushButton = QPushButton("Open MIDI file…")
        self.open_btn.clicked.connect(self.open_file)
        row.addWidget(self.open_btn)
        self.file_label = _named(QLabel("No file selected"), "fileinfo")
        row.addWidget(self.file_label, 1)
        return row

    def _build_settings_row(self) -> QHBoxLayout:
        """Build the size, frame rate and quality dropdowns plus the action buttons."""
        row = QHBoxLayout()
        row.setSpacing(8)

        self.res_box: QComboBox = QComboBox()
        self.res_box.addItems(list(RESOLUTIONS))
        self.res_box.setCurrentText("1080p")
        self.fps_box: QComboBox = QComboBox()
        self.fps_box.addItems(["24 fps", "30 fps", "60 fps"])
        self.fps_box.setCurrentText("30 fps")
        self.quality_box: QComboBox = QComboBox()
        self.quality_box.addItems(["high", "balanced", "fast"])

        for caption, widget in (("Size", self.res_box),
                                ("Frame rate", self.fps_box),
                                ("Quality", self.quality_box)):
            row.addWidget(_named(QLabel(caption), "subtitle"))
            row.addWidget(widget)
            row.addSpacing(6)
            # Changing a setting once a video exists rebuilds it straight away.
            widget.currentIndexChanged.connect(self._on_setting_changed)
        row.addStretch(1)      # push the buttons to the right

        self.cancel_btn: QPushButton = _named(QPushButton("Cancel"), "ghost")
        self.cancel_btn.clicked.connect(self._cancel_render)
        self.cancel_btn.hide()
        row.addWidget(self.cancel_btn)

        self.render_btn: QPushButton = _named(QPushButton("Save video"), "primary")
        self.render_btn.clicked.connect(self.create_video)
        self.render_btn.setEnabled(False)     # nothing to render yet
        row.addWidget(self.render_btn)
        return row

    def _build_progress_row(self) -> QHBoxLayout:
        """Build the render progress bar and its status text."""
        row = QHBoxLayout()
        row.setSpacing(10)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.hide()
        row.addWidget(self.progress, 1)
        self.status = _named(QLabel(), "status")
        self.status.setMinimumWidth(190)
        row.addWidget(self.status)
        return row

    # ------------------------------------------------------------------
    # Switching between the three views
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        """Show one of ``"empty"``, ``"preview"`` or ``"video"``."""
        self.mode = mode
        self.stack.setCurrentIndex(_PAGES[mode])

        has_media = mode in ("preview", "video")
        self.play_btn.setEnabled(has_media)
        self.slider.setEnabled(has_media)

        self.view_btn.setVisible(self.video_path is not None and mode != "empty")
        self.view_btn.setText("Back to preview" if mode == "video" else "Show video")

        # Only one of the two players may be heard at a time.
        if mode != "video":
            self.player.pause()
        if mode != "preview":
            self._stop_play()

    def _toggle_view(self) -> None:
        """Switch between the live preview and the finished video."""
        if self.mode == "video":
            self.player.pause()
            self._set_mode("preview")
            self._draw_preview()
        elif self.video_path:
            self._stop_play()
            self._set_mode("video")
            self.play_btn.setIcon(self.play_icon)

    # ------------------------------------------------------------------
    # Loading a file
    # ------------------------------------------------------------------

    def open_file(self) -> None:
        """Ask for a MIDI file and load it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open MIDI file", ensure_data_dir(),
            "MIDI files (*.mid *.midi);;All files (*)")
        if path:
            self.load_midi(path)

    def load_midi(self, path: str) -> None:
        """Read a MIDI file, prepare the preview, and show it.

        Anything that goes wrong -- a file that is not MIDI, or one with no
        notes -- is reported in a dialog, leaving whatever was loaded before.
        """
        self._stop_play()

        # Reading and preparing can take a moment, so show a busy cursor. On a
        # failure it goes back to normal before the dialog appears, not after.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            score = parse_midi(path)
            if score.note_count == 0:
                raise ValueError("This file contains no notes.")
            data = VisualData(score, fps=PREVIEW_FPS)
            renderer = FrameRenderer(data, PREVIEW_W, PREVIEW_H)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self, "Could not read file",
                f"{os.path.basename(path)} could not be loaded.\n\n"
                f"{type(exc).__name__}: {exc}")
            return
        QApplication.restoreOverrideCursor()

        self.midi_path = path
        self.score = score
        self.data = data
        self.renderer = renderer

        # Any video from a previous file no longer matches what is loaded.
        self.video_path = None
        self.player.setSource(QUrl())

        self.file_label.setText(self._describe(score))
        self.render_btn.setEnabled(True)
        self.status.setText("")
        self.progress.hide()

        self._preview_time = 0.0
        self._set_mode("preview")
        self._draw_preview()
        self._start_preview_audio()

    @staticmethod
    def _describe(score) -> str:
        """One line summarising a loaded score, for the label beside the Open button."""
        drum_tracks = sum(1 for t in score.tracks if t.is_drum)
        parts = [f"{len(score.tracks)} track{'s' if len(score.tracks) != 1 else ''}",
                 f"{score.note_count} notes",
                 f"{score.avg_bpm:.0f} BPM",
                 format_time(score.duration)]
        if drum_tracks:
            parts.insert(1, "1 drum kit" if drum_tracks == 1
                         else f"{drum_tracks} drum kits")
        return f"{score.name}   ·   " + "   ·   ".join(parts)

    # ------------------------------------------------------------------
    # The preview's sound
    # ------------------------------------------------------------------

    def _start_preview_audio(self) -> None:
        """Begin synthesizing the loaded score's sound in the background."""
        self._discard_preview_audio()
        if self.score is None:
            return
        self._set_status("Preparing sound…")
        # Synthesize with the same reverb a saved video would use, so the
        # result can be handed straight to the renderer later.
        reverb = RenderSettings().reverb
        self._pending_wav_source = (self.midi_path, reverb)
        self.audio_worker = AudioPreparer(self.score, reverb, self)
        self.audio_worker.ready.connect(self._on_preview_audio_ready)
        self.audio_worker.failed.connect(self._on_preview_audio_failed)
        self.audio_worker.start()

    def _discard_preview_audio(self) -> None:
        """Stop any sound in progress and throw away the file it was using."""
        if self.audio_worker is not None:
            self.audio_worker.cancel()
            self.audio_worker.wait(4000)
            self.audio_worker.deleteLater()
            self.audio_worker = None

        self.preview_player.stop()
        # The player keeps the file open, so let go before deleting it.
        self.preview_player.setSource(QUrl())
        _delete_quietly(self.preview_wav)
        self.preview_wav = None
        self.preview_wav_source = None
        self._preview_seek_ms = None

    def _on_preview_audio_ready(self, path: str) -> None:
        """Attach the finished sound to the preview.

        The sound usually arrives while the picture is already part way
        through, so it has to start from wherever the picture has reached. A
        player will not accept a position until it has finished opening the
        file, so the request is remembered and applied by
        :meth:`_on_preview_media_status`. Seeking here instead would be thrown
        away, the sound would start from zero, and -- because the picture
        follows the sound -- the animation would jump backwards.
        """
        if self.sender() is not self.audio_worker:
            # A leftover from a file that has since been replaced.
            _delete_quietly(path)
            return

        self.preview_wav = path
        self.preview_wav_source = self._pending_wav_source
        self._preview_seek_ms = int(self._preview_time * 1000)
        self.preview_player.setSource(QUrl.fromLocalFile(path))
        self._set_status("")

    def _on_preview_media_status(self, status) -> None:
        """Start the preview's sound at the right moment, once it has loaded."""
        if self._preview_seek_ms is None:
            return
        ready = (QMediaPlayer.MediaStatus.LoadedMedia,
                 QMediaPlayer.MediaStatus.BufferingMedia,
                 QMediaPlayer.MediaStatus.BufferedMedia)
        if status not in ready or self.preview_player.duration() <= 0:
            return

        self.preview_player.setPosition(self._preview_seek_ms)
        self._preview_seek_ms = None
        # Catch up with the picture if it is already running.
        if self._playing:
            self.preview_player.play()

    def _on_preview_audio_failed(self, message: str) -> None:
        """Carry on without sound, saying so rather than failing silently."""
        self.preview_wav = None
        self._set_status("Preview has no sound: " + message)

    def _on_volume_changed(self, value: int) -> None:
        """Set how loud playback is in the app, for both players.

        Loudness is not heard in a straight line: halfway along the slider
        should sound like half as loud, which is not the same as half the
        signal level. The logarithmic curve below makes that translation. It is
        the one Qt's ``QAudio.convertVolume`` applies, written out here because
        PyQt's type stubs declare that function incorrectly.

        This changes nothing about a saved video. That file's audio is
        synthesized separately by the pipeline and always written at full level.
        """
        fraction = value / 100.0
        linear = 1.0 if fraction > 0.99 else -math.log(1.0 - fraction) / math.log(100.0)
        self.audio_out.setVolume(linear)
        self.preview_audio_out.setVolume(linear)

    def _set_status(self, text: str) -> None:
        """Update the status line, unless a render is using it."""
        if self.worker is None:
            self.status.setText(text)

    def _has_preview_sound(self) -> bool:
        """True once the preview's sound is loaded and can act as the clock.

        Having the file is not enough: until the player has opened it and
        applied the position it was asked for, it reports a position of zero.
        Treating that as the clock would drag the animation back to the start.
        """
        return (self.preview_wav is not None
                and self._preview_seek_ms is None      # the jump has been applied
                and self.preview_player.duration() > 0)

    def _move_preview_sound_to(self, seconds: float) -> None:
        """Point the preview's sound at a moment, now or once it has loaded."""
        if self.preview_wav is None:
            return
        milliseconds = int(seconds * 1000)
        if self._preview_seek_ms is not None:
            # Still opening the file, so revise the jump it will make.
            self._preview_seek_ms = milliseconds
        else:
            self.preview_player.setPosition(milliseconds)

    # ------------------------------------------------------------------
    # The animation preview
    # ------------------------------------------------------------------

    def _draw_preview(self) -> None:
        """Render the frame at the current preview time and update the transport."""
        if self.renderer is None or self.data is None:
            return
        frame = self.data.frame_at_time(self._preview_time)
        self.canvas.set_image(self.renderer.render(frame))

        # Do not fight the user for the slider while they are dragging it.
        if not self.slider.isSliderDown():
            self.slider.setValue(
                int(1000 * self._preview_time / max(self.data.duration, 1e-6)))
        self.time_label.setText(
            f"{format_time(self._preview_time)} / {format_time(self.data.duration)}")

    def _advance_preview(self) -> None:
        """Move the preview forward. Called by the timer while playing.

        When the sound is ready it is the clock: the picture follows the audio
        player's position, so the two can never drift apart. Until then the
        elapsed-time fallback keeps real time on its own.
        """
        if self.mode != "preview" or self.data is None:
            return

        if self._has_preview_sound():
            self._preview_time = self.preview_player.position() / 1000.0
        else:
            self._preview_time += self._clock.restart() / 1000.0

        if self._preview_time >= self.data.duration:
            self._preview_time = self.data.duration
            self._stop_play()
        self._draw_preview()

    def _toggle_play(self) -> None:
        """Play or pause, whichever applies to the view currently showing."""
        if self.mode == "video":
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            else:
                self.player.play()
            return

        if self.mode != "preview" or self.data is None:
            return
        if self._playing:
            self._stop_play()
            return

        # Starting from the very end means the user wants to watch it again.
        if self._preview_time >= self.data.duration - 0.05:
            self._preview_time = 0.0

        self._playing = True
        self.play_btn.setIcon(self.pause_icon)
        self._clock.restart()
        self._timer.start()
        self._move_preview_sound_to(self._preview_time)
        if self._has_preview_sound():
            self.preview_player.play()
        # If the sound is still loading, _on_preview_media_status starts it.

    def _stop_play(self) -> None:
        """Pause the preview, picture and sound together."""
        self._playing = False
        self._timer.stop()
        self.play_btn.setIcon(self.play_icon)
        self.preview_player.pause()

    def _on_scrub_start(self) -> None:
        """Pause while the slider is held, remembering whether to resume.

        Letting playback continue during a drag would fight the user for the
        handle, but stopping for good is not what a media player does either:
        whatever was playing before carries on once the handle is released.
        """
        self._resume_after_scrub = (self._playing if self.mode == "preview"
                                    else self._video_is_playing())
        if self.mode == "preview":
            self._stop_play()
        else:
            self.player.pause()

    def _on_scrub(self, value: int) -> None:
        """Jump to the position the slider was moved to, on a scale of 0-1000."""
        if self.mode == "video":
            if self.player.duration() > 0:
                self.player.setPosition(int(self.player.duration() * value / 1000))
        elif self.data is not None:
            self._preview_time = self.data.duration * value / 1000.0
            self._move_preview_sound_to(self._preview_time)
            self._draw_preview()

    def _on_scrub_end(self) -> None:
        """Carry on playing after a drag, if it was playing before."""
        if not self._resume_after_scrub:
            return
        self._resume_after_scrub = False
        if self.mode == "preview":
            if not self._playing:
                self._toggle_play()
        else:
            self.player.play()

    def _video_is_playing(self) -> bool:
        """True when the finished video is playing."""
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    # ------------------------------------------------------------------
    # The video player
    # ------------------------------------------------------------------

    def _on_player_position(self, position_ms: int) -> None:
        """Keep the slider and clock in step with the playing video."""
        if self.mode != "video":
            return
        duration = max(1, self.player.duration())
        if not self.slider.isSliderDown():
            self.slider.setValue(int(1000 * position_ms / duration))
        self.time_label.setText(
            f"{format_time(position_ms / 1000)} / {format_time(duration / 1000)}")

    def _on_player_duration(self, duration_ms: int) -> None:
        """Show the total time as soon as the player has worked it out."""
        if self.mode == "video":
            self.time_label.setText(f"0:00 / {format_time(duration_ms / 1000)}")
        # Knowing the length may be the last thing a pending resume was waiting for.
        self._try_resume()

    def _on_player_state(self, state) -> None:
        """Keep the play button's icon matching the player."""
        if self.mode == "video":
            playing = state == QMediaPlayer.PlaybackState.PlayingState
            self.play_btn.setIcon(self.pause_icon if playing else self.play_icon)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _settings(self) -> RenderSettings:
        """Read the dropdowns into a :class:`RenderSettings`."""
        return RenderSettings(
            resolution=self.res_box.currentText(),
            fps=int(self.fps_box.currentText().split()[0]),   # "30 fps" -> 30
            quality=self.quality_box.currentText(),
        )

    def create_video(self) -> None:
        """Ask where to save, then start the render on a worker thread."""
        if not self.midi_path or self.worker is not None:
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save video as", default_output_path(self.midi_path),
            "MP4 video (*.mp4)")
        if not out_path:
            return
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"

        self._stop_play()
        # Let go of any file the player has open: Windows will not allow
        # overwriting a file that is still in use.
        self.player.setSource(QUrl())
        self.video_path = None
        # A fresh save always starts from the beginning.
        self._resume_after_render = None
        self._resume_ms = None
        self._resume_playing = False

        self.status.setText("Starting…")
        self._start_render(out_path)

    def _reusable_audio(self) -> str | None:
        """The preview's sound file, if a saved video may be built from it.

        The preview has already synthesized this MIDI, and a saved video needs
        exactly the same audio: resolution and frame rate change the picture
        only. Handing the existing file to the renderer skips the synthesis
        step, which is roughly a third of the work.

        It is offered only when the file was made from the MIDI now loaded and
        with the same reverb the render will use, so a stale or
        differently-voiced recording can never reach a saved video.

        Returns:
            A path to reuse, or None to synthesize afresh.
        """
        wanted = (self.midi_path, self._settings().reverb)
        if self.preview_wav and self.preview_wav_source == wanted:
            return self.preview_wav
        return None

    def _start_render(self, out_path: str) -> None:
        """Run the conversion on a worker thread, writing to ``out_path``.

        Shared by saving for the first time and by rebuilding after a setting
        changed, so both behave identically.
        """
        self._set_busy(True)
        self.progress.setValue(0)
        self.progress.show()

        self.worker = RenderWorker(self.midi_path, out_path, self._settings(),
                                   self._reusable_audio(), self)
        self.worker.progressed.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_render_done)
        self.worker.failed.connect(self._on_render_failed)
        self.worker.start()          # calls RenderWorker.run() on a new thread

    def _on_setting_changed(self) -> None:
        """Rebuild the video being watched when a setting changes.

        This only happens while the finished video is on screen, which is the
        moment the change is worth acting on immediately. Anywhere else -- with
        nothing saved yet, or while looking at the preview -- the new setting
        simply applies the next time Save video is pressed, rather than
        starting a long render nobody asked for and switching the view away.
        """
        if self._rerender_ready():
            self._rerender_video()

    def _rerender_ready(self) -> bool:
        """True when the finished video is on screen and could be rebuilt now."""
        return (self.video_path is not None
                and self.worker is None
                and self.mode == "video")

    def _rerender_video(self) -> None:
        """Re-render the saved video with the current settings, then resume.

        The position and whether it was playing are remembered first, so that
        when the new file is ready it picks up from the same moment.
        """
        if not self._rerender_ready():
            return

        self._resume_after_render = (self.player.position(), self._video_is_playing())

        self.player.pause()
        # Windows will not let ffmpeg overwrite a file the player still holds.
        self.player.setSource(QUrl())

        self._set_status("Applying new settings…")
        self._start_render(self.video_path)

    def _set_busy(self, busy: bool) -> None:
        """Disable the controls that must not change mid-render."""
        for widget in (self.open_btn, self.render_btn, self.res_box,
                       self.fps_box, self.quality_box):
            widget.setEnabled(not busy)
        if not busy and self.midi_path:
            self.render_btn.setEnabled(True)
        self.cancel_btn.setVisible(busy)

    def _on_progress(self, fraction: float, label: str) -> None:
        """Show how far along the render is."""
        self.progress.setValue(int(fraction * 1000))
        self.status.setText(f"{label}   {int(fraction * 100)}%")

    def _cancel_render(self) -> None:
        """Ask the worker to stop. It finishes the frame it is on first."""
        if self.worker is not None:
            self.status.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)
            self.worker.cancel()

    def _clear_worker(self) -> None:
        """Wait for the worker to finish and let the controls work again."""
        if self.worker is not None:
            self.worker.wait(3000)
            # Qt deletes the object once it is safe to do so.
            self.worker.deleteLater()
            self.worker = None
        self._set_busy(False)
        self.cancel_btn.setEnabled(True)

    def _on_render_done(self, path: str) -> None:
        """Show the finished video and start playing it."""
        rebuilt = self._resume_after_render is not None
        self._clear_worker()
        self.video_path = path
        self.progress.setValue(1000)
        self.status.setText("Updated" if rebuilt else "Done")
        QTimer.singleShot(1600, self.progress.hide)

        if rebuilt:
            # Arm the resume now, so it can only be applied to the new file.
            self._resume_ms, self._resume_playing = self._resume_after_render
            self._resume_after_render = None

        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(path)))
        self._set_mode("video")
        if not rebuilt:
            self.player.play()
        # A rebuild instead resumes from _try_resume, once the file has loaded.

        folder = os.path.dirname(os.path.abspath(path))
        self.file_label.setText(f"{os.path.basename(path)}   ·   saved to {folder}")
        if not rebuilt:
            reveal_in_file_manager(path)

    def _on_media_status(self, status) -> None:
        """Try to restore the playback position when the media becomes ready."""
        ready = (QMediaPlayer.MediaStatus.LoadedMedia,
                 QMediaPlayer.MediaStatus.BufferedMedia)
        if status in ready:
            self._try_resume()

    def _try_resume(self) -> None:
        """Jump back to the remembered moment, once that is actually possible.

        A player cannot seek in a file it has not finished opening, and it does
        not know the length straight away either. Both the "media is ready" and
        the "length is known" signals call this, and it does nothing until both
        have happened -- seeking before then would be clamped to zero and the
        video would restart from the beginning.
        """
        if self._resume_ms is None:
            return

        # The length is often announced while the file is still opening, and a
        # seek made then is thrown away when loading finishes. Both the length
        # and a loaded file are required before it will stick.
        duration = self.player.duration()
        loaded = (QMediaPlayer.MediaStatus.LoadedMedia,
                  QMediaPlayer.MediaStatus.BufferingMedia,
                  QMediaPlayer.MediaStatus.BufferedMedia)
        if duration <= 0 or self.player.mediaStatus() not in loaded:
            return

        # Stop just short of the end, so resuming never lands past the finish.
        self.player.setPosition(min(self._resume_ms, max(0, duration - 200)))
        if self._resume_playing:
            self.player.play()
        self._resume_ms = None
        self._resume_playing = False

    def _on_render_failed(self, kind: str, message: str) -> None:
        """Report a cancelled or failed render.

        If a rebuild was interrupted the previous file is still on disk, so it
        is put back in the player rather than leaving the window empty.
        """
        was_rebuild = self._resume_after_render is not None
        self._clear_worker()
        self.progress.hide()

        if was_rebuild and self.video_path and os.path.isfile(self.video_path):
            # The previous file survived, so put it back rather than leaving
            # the window blank, and return to where the user was.
            self._resume_ms, self._resume_playing = self._resume_after_render
            self.player.setSource(QUrl.fromLocalFile(os.path.abspath(self.video_path)))
        self._resume_after_render = None

        if kind == "cancelled":
            self.status.setText("Cancelled")
            return

        self.status.setText("Failed")
        if kind == "ffmpeg":
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("ffmpeg is required")
            box.setText("The video could not be written because ffmpeg is missing.")
            box.setInformativeText(message)
            box.exec()
        else:
            QMessageBox.critical(self, "Rendering failed", message)

    # ------------------------------------------------------------------
    # Dragging a file onto the window
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        """Accept the drag only if it carries a MIDI file and we are not busy."""
        if event.mimeData().hasUrls() and self._midi_from(event) and self.worker is None:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        """Load the MIDI file that was dropped."""
        path = self._midi_from(event)
        if path:
            event.acceptProposedAction()
            self.load_midi(path)

    @staticmethod
    def _midi_from(event) -> str | None:
        """Return the first MIDI file in a drag event, or None."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".mid", ".midi")):
                return path
        return None

    def closeEvent(self, event) -> None:
        """Stop the render and playback cleanly when the window is closed."""
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(4000)
        self._timer.stop()
        self.player.stop()
        # Also stops the sound thread and deletes its temporary file.
        self._discard_preview_audio()
        super().closeEvent(event)


def run(midi_path: str | None = None) -> int:
    """Open the window and run until it is closed.

    Args:
        midi_path: a file to load straight away, if one was named.

    Returns:
        The process exit code.
    """
    import sys

    # The Qt media backend prints codec chatter to stderr; keep the console clean.
    os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia.*=false")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("MIDI Visualizer")

    # Tidy away sound files from any session that did not close cleanly.
    _sweep_stale_preview_audio()

    window = MainWindow()
    window.show()

    if midi_path and os.path.isfile(midi_path):
        # Load once the window is on screen, so any error dialog has a parent.
        QTimer.singleShot(60, lambda: window.load_midi(midi_path))

    return app.exec()
