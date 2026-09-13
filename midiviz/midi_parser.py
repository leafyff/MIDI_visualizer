"""Reads a MIDI file and turns it into a plain ``Score`` object.

This is the only module that knows about the ``mido`` library or about MIDI's
quirks. Everything downstream (the synthesizer, the animation) works on the
simple dataclasses defined here, where every time is a float in **seconds**.

A short primer on the MIDI facts this module has to deal with:

* A MIDI file stores time in **ticks**, not seconds. How long a tick lasts
  depends on the current tempo, which can change part-way through the song, so
  converting ticks to seconds needs the tempo map built by ``_TempoMap``.
* A note is two separate events: ``note_on`` and ``note_off``. We have to pair
  them up ourselves. By a very common convention a ``note_on`` with velocity 0
  also means "note off".
* **Channel 9** (the tenth, counting from zero) is always percussion. On that
  channel a note number selects a drum instead of a pitch.
* Instruments can be laid out in two ways, and we support both -- see
  :func:`parse_midi`.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field

import mido

from . import gm

#: Tempo MIDI assumes when a file specifies none: 500000 microseconds per beat,
#: which works out to 120 BPM.
DEFAULT_TEMPO = 500_000

#: The drum channel, counting from zero (so it is "channel 10" in most software).
DRUM_CHANNEL = 9

#: One colour per track, chosen to stay distinguishable side by side. A file may
#: hold up to ten tracks; beyond that the palette simply repeats.
PALETTE = [
    (255, 92, 108),    # red
    (255, 168, 61),    # amber
    (255, 233, 84),    # yellow
    (126, 231, 135),   # green
    (68, 214, 196),    # teal
    (77, 171, 255),    # blue
    (140, 130, 255),   # indigo
    (200, 116, 255),   # violet
    (255, 110, 199),   # pink
    (170, 204, 120),   # olive
]


@dataclass(slots=True)
class Note:
    """A single note, with its timing already converted to seconds."""

    start: float      # when the note begins, in seconds from the start of the song
    end: float        # when the note is released, in seconds
    pitch: int        # MIDI note number, 0-127 (60 is middle C)
    velocity: int     # how hard the note was struck, 1-127
    track: int        # index of the owning track in Score.tracks

    @property
    def duration(self) -> float:
        """How long the note is held, in seconds."""
        return self.end - self.start


@dataclass
class Track:
    """One instrument: every note that shares a track and a MIDI channel."""

    index: int            # position in Score.tracks; also picks the colour
    name: str             # label shown in the video, e.g. "Bass"
    program: int          # General MIDI instrument number, 0-127
    channel: int          # MIDI channel the notes came from, 0-15
    is_drum: bool         # True when channel == DRUM_CHANNEL
    notes: list[Note] = field(default_factory=list)
    volume: float = 1.0   # from MIDI controller 7, as 0.0-1.0
    pan: float = 0.0      # from MIDI controller 10, as -1.0 (left) to +1.0 (right)

    @property
    def color(self) -> tuple[int, int, int]:
        """The track's RGB colour, taken from :data:`PALETTE`."""
        return PALETTE[self.index % len(PALETTE)]

    @property
    def family(self) -> str:
        """Instrument family name, e.g. ``"bass"`` -- the synth picks a tone from it."""
        return gm.family_of(self.program, self.is_drum)


@dataclass
class Score:
    """A whole song: its tracks plus the timing information drawn from it."""

    name: str                                 # the file name, shown in the video
    tracks: list[Track]
    duration: float                           # seconds, including a short tail
    beats: list[float]                        # time of every beat, in seconds
    downbeats: list[float]                    # time of the first beat of each bar
    tempo_changes: list[tuple[float, float]]  # (seconds, BPM) at each tempo change

    @property
    def note_count(self) -> int:
        """Total number of notes across every track."""
        return sum(len(t.notes) for t in self.tracks)

    @property
    def avg_bpm(self) -> float:
        """Average tempo, used to set the speed of the animation."""
        if not self.tempo_changes:
            return 120.0
        return sum(bpm for _, bpm in self.tempo_changes) / len(self.tempo_changes)

    def pitch_bounds(self) -> tuple[int, int]:
        """Lowest and highest pitch used, as ``(low, high)``.

        Falls back to a sensible middle range when the score has no notes, so
        callers never have to special-case an empty song.
        """
        pitches = [n.pitch for t in self.tracks for n in t.notes]
        if not pitches:
            return 36, 84
        return min(pitches), max(pitches)


class _TempoMap:
    """Converts tick positions into seconds, following every tempo change.

    MIDI tempo is given as microseconds per beat. Between two tempo changes the
    rate is constant, so we record the running time in seconds at each change
    and interpolate linearly from the nearest one at or before the tick asked
    for.
    """

    def __init__(self, events: list[tuple[int, int]], ticks_per_beat: int):
        """
        Args:
            events: ``(tick, microseconds_per_beat)`` pairs, sorted by tick.
            ticks_per_beat: the file's tick resolution, from the MIDI header.
        """
        self.ticks_per_beat = max(1, ticks_per_beat)

        # There must always be a tempo in force at tick 0, so insert the MIDI
        # default if the file does not set one itself.
        if not events or events[0][0] != 0:
            events = [(0, DEFAULT_TEMPO)] + list(events)

        self.ticks: list[int] = []
        self.seconds: list[float] = []
        self.tempos: list[int] = []

        elapsed = 0.0
        prev_tick, prev_tempo = 0, events[0][1]
        for tick, tempo in events:
            elapsed += self._span(tick - prev_tick, prev_tempo)
            self.ticks.append(tick)
            self.seconds.append(elapsed)
            self.tempos.append(tempo)
            prev_tick, prev_tempo = tick, tempo

    def _span(self, ticks: int, tempo: int) -> float:
        """How many seconds ``ticks`` last at ``tempo`` microseconds per beat."""
        return ticks * (tempo / 1_000_000.0) / self.ticks_per_beat

    def to_seconds(self, tick: int) -> float:
        """Convert an absolute tick position to seconds."""
        # bisect finds where tick would be inserted; the entry before it is the
        # last tempo change at or before this tick.
        i = max(0, bisect.bisect_right(self.ticks, tick) - 1)
        return self.seconds[i] + self._span(tick - self.ticks[i], self.tempos[i])


@dataclass
class _Group:
    """Notes and settings gathered for one ``(track, channel)`` pair.

    This is scratch data used while parsing; it becomes a :class:`Track` once
    the tick positions have been converted to seconds.
    """

    # Raw notes as (start_tick, end_tick, pitch, velocity), still in ticks.
    raw: list[tuple[int, int, int, int]] = field(default_factory=list)
    name: str = ""
    program: int = 0
    volume: float = 1.0
    pan: float = 0.0


def _read_file(mid: mido.MidiFile):
    """Walk every message once, converting delta times to absolute ticks.

    MIDI stores each message's time as a delta from the previous one, which is
    awkward to work with, so we add them up as we go.

    Returns:
        ``(per_track, tempo_map, time_signatures)`` where ``per_track`` is a
        list (one entry per MIDI track) of ``(tick, message)`` pairs, and
        ``time_signatures`` is a sorted list of ``(tick, beats_per_bar)``.
    """
    per_track: list[list[tuple[int, mido.Message]]] = []
    tempo_events: dict[int, int] = {}          # tick -> microseconds per beat
    time_signatures: list[tuple[int, int]] = []  # (tick, beats per bar)

    for track in mid.tracks:
        tick = 0
        messages = []
        for msg in track:
            tick += msg.time
            # Tempo and time signature are global, so collect them from every
            # track -- most files put them in the first one, but not all.
            if msg.type == "set_tempo":
                tempo_events[tick] = msg.tempo
            elif msg.type == "time_signature":
                time_signatures.append((tick, msg.numerator))
            messages.append((tick, msg))
        per_track.append(messages)

    tempo_map = _TempoMap(sorted(tempo_events.items()), mid.ticks_per_beat)

    time_signatures.sort()
    if not time_signatures or time_signatures[0][0] != 0:
        time_signatures.insert(0, (0, 4))      # assume 4/4 until told otherwise
    return per_track, tempo_map, time_signatures


def _repair_text(text: str) -> str:
    """Undo the mangling that happens when UTF-8 text is read as Latin-1.

    MIDI never settled on a text encoding, so ``mido`` decodes names as Latin-1,
    where every byte is a character and nothing can fail. A file that actually
    used UTF-8 therefore arrives scrambled: "B♭ Clarinets" comes out as
    "Bâ\\x99\\xad Clarinets".

    Encoding back to the original bytes and decoding them as UTF-8 recovers the
    real text. If that fails, the name genuinely was Latin-1 and is returned
    untouched.
    """
    if text.isascii():
        return text                  # nothing outside ASCII, so nothing to repair
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _label_for(track_name: str, program: int, is_drum: bool, shares_track: bool) -> str:
    """Build the name shown next to a track's bar in the video.

    Args:
        track_name: the file's own name for the MIDI track, if it gave one.
        program: General MIDI instrument number.
        is_drum: whether this is the percussion channel.
        shares_track: True when several channels live on the same MIDI track, in
            which case the track's name alone would not tell them apart.
    """
    track_name = (track_name or "").strip()
    instrument = gm.program_name(program, is_drum)
    if not track_name:
        return instrument
    if shares_track:
        return f"{track_name} ({instrument})"
    return track_name


def _collect_groups(per_track, ticks_per_beat: int) -> dict[tuple[int, int], _Group]:
    """Pair note-ons with note-offs and sort the result by track and channel.

    Returns:
        A dict keyed by ``(track_index, channel)``. Python keeps dict keys in
        insertion order, so the tracks come out in the order the file lists them.
    """
    groups: dict[tuple[int, int], _Group] = {}

    for track_index, messages in enumerate(per_track):
        track_name = ""
        programs: dict[int, int] = {}     # channel -> instrument number
        volumes: dict[int, float] = {}    # channel -> controller 7
        pans: dict[int, float] = {}       # channel -> controller 10
        channels_seen: set[int] = set()

        # Notes waiting for their note-off, keyed by (channel, pitch). It is a
        # list because the same pitch can legitimately be struck again before
        # the first one is released; the oldest is released first.
        pending: dict[tuple[int, int], list[tuple[int, int]]] = {}

        def group_for(channel_: int) -> _Group:
            """Fetch (or create) the group for this track and channel."""
            return groups.setdefault((track_index, channel_), _Group())

        for tick, msg in messages:
            if msg.type == "track_name":
                track_name = track_name or _repair_text(msg.name)
            elif msg.type == "instrument_name" and not track_name:
                track_name = _repair_text(msg.name)
            elif msg.type == "program_change":
                programs[msg.channel] = msg.program
            elif msg.type == "control_change":
                # Keep the first setting of each: later ones are usually
                # mid-song automation, which we do not model.
                if msg.control == 7:
                    volumes.setdefault(msg.channel, msg.value / 127.0)
                elif msg.control == 10:
                    pans.setdefault(msg.channel, (msg.value - 64) / 63.0)
            elif msg.type == "note_on" and msg.velocity > 0:
                channels_seen.add(msg.channel)
                pending.setdefault((msg.channel, msg.note), []).append((tick, msg.velocity))
            elif msg.type == "note_off" or msg.type == "note_on":
                # A note_on that reaches here has velocity 0, which means off.
                channels_seen.add(msg.channel)
                waiting = pending.get((msg.channel, msg.note))
                if waiting:
                    start_tick, velocity = waiting.pop(0)
                    group_for(msg.channel).raw.append(
                        (start_tick, tick, msg.note, velocity))

        # Some files forget the final note-offs. Rather than dropping those
        # notes, end them at the track's last event (or after a beat, whichever
        # is later) so they still appear and sound.
        last_tick = messages[-1][0] if messages else 0
        for (channel, pitch), waiting in pending.items():
            for start_tick, velocity in waiting:
                end_tick = max(last_tick, start_tick + ticks_per_beat)
                group_for(channel).raw.append((start_tick, end_tick, pitch, velocity))

        # Now that every channel on this track is known, fill in the settings.
        shares_track = len(channels_seen) > 1
        for channel in channels_seen:
            group = groups.get((track_index, channel))
            if group is None:
                continue
            group.program = programs.get(channel, 0)
            group.volume = volumes.get(channel, 1.0)
            group.pan = pans.get(channel, 0.0)
            group.name = _label_for(track_name, group.program,
                                    channel == DRUM_CHANNEL, shares_track)

    return groups


def _build_tracks(groups: dict[tuple[int, int], _Group],
                  tempo_map: _TempoMap) -> list[Track]:
    """Turn parsing scratch data into :class:`Track` objects with times in seconds."""
    tracks: list[Track] = []

    for (_, channel), group in groups.items():
        if not group.raw:
            continue
        index = len(tracks)
        track = Track(
            index=index,
            name=group.name or gm.program_name(group.program, channel == DRUM_CHANNEL),
            program=group.program,
            channel=channel,
            is_drum=(channel == DRUM_CHANNEL),
            volume=group.volume,
            pan=group.pan,
        )
        for start_tick, end_tick, pitch, velocity in group.raw:
            start = tempo_map.to_seconds(start_tick)
            end = tempo_map.to_seconds(end_tick)
            # Guard against zero-length notes, which would be silent and invisible.
            track.notes.append(Note(start, max(end, start + 0.03),
                                    int(pitch), int(velocity), index))
        track.notes.sort(key=lambda n: (n.start, n.pitch))
        tracks.append(track)

    return tracks


def _beat_grid(tempo_map: _TempoMap, time_signatures: list[tuple[int, int]],
               last_tick: int, ticks_per_beat: int,
               duration: float) -> tuple[list[float], list[float]]:
    """List the time of every beat, and of every bar's first beat.

    The animation flashes on these, harder on a downbeat.

    Returns:
        ``(beats, downbeats)``, both in seconds.
    """
    beats: list[float] = []
    downbeats: list[float] = []

    tick = 0
    signature_index = 0
    beats_per_bar = time_signatures[0][1]
    position_in_bar = 0

    # Step one beat at a time. The step count is bounded so that a corrupt file
    # claiming a huge length cannot spin here forever.
    for _ in range(200_000):
        if tick > max(last_tick, 1):
            break
        # Adopt a new time signature once we reach it.
        while (signature_index + 1 < len(time_signatures)
               and time_signatures[signature_index + 1][0] <= tick):
            signature_index += 1
            beats_per_bar = time_signatures[signature_index][1]
            position_in_bar = 0

        seconds = tempo_map.to_seconds(tick)
        if seconds > duration:
            break
        beats.append(seconds)
        if position_in_bar == 0:
            downbeats.append(seconds)

        position_in_bar = (position_in_bar + 1) % max(1, beats_per_bar)
        tick += ticks_per_beat

    return beats, downbeats


def parse_midi(path: str) -> Score:
    """Read a MIDI file and return it as a :class:`Score`.

    Notes are grouped by ``(track, channel)`` pairs, which handles both ways a
    file can be laid out:

    * a **type 1** file usually puts one instrument on each MIDI track;
    * a **type 0** file puts everything on a single track and separates the
      instruments by channel instead.

    Grouping on the pair covers both without needing to know which kind we got.

    Args:
        path: path to a ``.mid`` or ``.midi`` file.

    Returns:
        The parsed score. A file with no notes yields a Score with no tracks --
        callers decide whether that is an error.

    Raises:
        OSError: if the file is missing or is not a MIDI file at all.
    """
    mid = mido.MidiFile(path)
    per_track, tempo_map, time_signatures = _read_file(mid)

    groups = _collect_groups(per_track, mid.ticks_per_beat)
    tracks = _build_tracks(groups, tempo_map)

    # Run on a little past the final note so its tail is not cut off.
    last_note_end = max((n.end for t in tracks for n in t.notes), default=0.0)
    duration = max(last_note_end + 1.5, 2.0)

    last_tick = max((msgs[-1][0] for msgs in per_track if msgs), default=0)
    beats, downbeats = _beat_grid(tempo_map, time_signatures, last_tick,
                                  mid.ticks_per_beat, duration)

    # Microseconds per beat -> beats per minute.
    tempo_changes = [(seconds, 60_000_000.0 / tempo)
                     for seconds, tempo in zip(tempo_map.seconds, tempo_map.tempos)]

    return Score(
        name=os.path.basename(path),
        tracks=tracks,
        duration=duration,
        beats=beats,
        downbeats=downbeats,
        tempo_changes=tempo_changes,
    )
