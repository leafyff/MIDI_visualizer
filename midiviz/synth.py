"""Turns a Score into audio. A small software synthesizer built on numpy.

Why this exists: ffmpeg cannot decode MIDI, and we cannot assume the machine has
FluidSynth or a SoundFont installed. So the program makes its own sound.

How it works, in three ideas:

1. **Wavetables.** Instead of computing sine waves for every note (slow), we
   build one cycle of each instrument's waveform once, then play notes by
   stepping through that table at the right speed. Reading a table is just an
   array lookup, so rendering thousands of notes stays fast.

2. **Band limiting.** A bright waveform such as a sawtooth is a stack of
   harmonics. Any harmonic above half the sample rate (the *Nyquist* frequency,
   22050 Hz here) cannot be represented and folds back as an unmusical whine --
   *aliasing*. To avoid it we build one table per octave, each containing only
   the harmonics that still fit for notes in that octave.

3. **Drums are different.** Percussion has no pitch, so channel 10 notes are
   built from shaped noise plus a short pitch-swept sine, described by the
   :data:`DRUM_KIT` table.

The public entry points are :func:`render_score` and :func:`write_wav`.
"""

from __future__ import annotations

import math
import wave
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .midi_parser import Score

#: Audio samples per second. 44100 is CD quality.
SAMPLE_RATE = 44100

#: Samples in one cycle of a wavetable. A power of two keeps the maths tidy.
TABLE_SIZE = 2048

#: How many octave-sized wavetables to build (MIDI pitches 0-127 span 11).
OCTAVES = 11

#: Frequencies above this would alias, so no harmonic is allowed past it.
#: Slightly under the true Nyquist limit to leave a little headroom.
MAX_FREQUENCY = SAMPLE_RATE * 0.45


def midi_to_freq(pitch: float) -> float:
    """Convert a MIDI note number to a frequency in Hz (note 69 = A4 = 440 Hz)."""
    return 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))


# --------------------------------------------------------------------------
# Instrument tone definitions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Timbre:
    """How one family of instruments sounds.

    The first four fields are a classic **ADSR envelope**, which shapes a note's
    loudness over time:

    * ``attack``  -- seconds to fade in from silence when the note starts;
    * ``decay``   -- how quickly it falls from full volume towards ``sustain``;
    * ``sustain`` -- the level it settles at while the key is held, 0.0-1.0;
    * ``release`` -- how quickly it fades out after the key is let go.

    A piano has a near-instant attack and no sustain (it rings then dies); a
    string pad fades in slowly and holds. That difference is most of what makes
    instruments recognisable.
    """

    attack: float
    decay: float
    sustain: float
    release: float
    gain: float           # overall loudness of this family, relative to the others
    voices: int = 1       # how many slightly detuned copies to stack, for richness
    detune: float = 0.0   # spread between those copies, in cents (100 = a semitone)
    spectrum: str = "saw"  # which waveform recipe in _harmonic_amplitude to use


def _harmonic_amplitude(spectrum: str, h: int) -> float:
    """How loud harmonic number ``h`` is in a given waveform.

    Harmonic 1 is the note you hear as the pitch; 2 is an octave above it, and
    so on. The pattern of their volumes is what makes a waveform sound bright or
    mellow -- a sawtooth (every harmonic, fading as 1/h) sounds buzzy, while a
    sine (harmonic 1 alone) sounds pure.
    """
    if spectrum == "sine":
        return 1.0 if h == 1 else 0.0
    if spectrum == "saw":
        return 1.0 / h
    if spectrum == "triangle":      # odd harmonics only, falling away quickly
        if h % 2 == 0:
            return 0.0
        return (1.0 / (h * h)) * (1.0 if h % 4 == 1 else -1.0)
    if spectrum == "piano":         # bright, with the even harmonics held back
        return (1.0 / (h ** 1.35)) * (1.0 if h % 2 else 0.72)
    if spectrum == "organ":         # a fixed set of drawbars
        return {1: 1.0, 2: 0.62, 3: 0.35, 4: 0.42, 6: 0.18, 8: 0.24}.get(h, 0.0)
    if spectrum == "bell":          # sparse and metallic
        return {1: 1.0, 2: 0.28, 3: 0.62, 4: 0.34,
                5: 0.45, 7: 0.3, 9: 0.22, 11: 0.16}.get(h, 0.0)
    if spectrum == "pluck":
        return 1.0 / (h ** 1.15)
    if spectrum == "brass":         # strong low harmonics, then a sharp drop
        return (1.0 / (h ** 0.85)) if h <= 12 else (1.0 / (h ** 1.9))
    if spectrum == "reed":          # odd harmonics lead, like a clarinet
        return (1.0 / (h ** 1.05)) if h % 2 else (0.35 / (h ** 1.3))
    if spectrum == "soft":          # mellow, harmonics fade fast
        return 1.0 / (h ** 1.8)
    return 1.0 / h


#: One tone per instrument family. Family names come from ``gm.family_of()``.
FAMILY_TIMBRE: dict[str, Timbre] = {
    #                attack decay sustain release gain voices detune spectrum
    "piano":      Timbre(0.004, 1.10, 0.18, 0.28, 0.95, 1, 0.0, "piano"),
    "chromatic":  Timbre(0.002, 0.70, 0.00, 0.45, 0.85, 1, 0.0, "bell"),
    "organ":      Timbre(0.020, 0.06, 0.95, 0.10, 0.70, 2, 4.0, "organ"),
    "guitar":     Timbre(0.004, 1.30, 0.12, 0.25, 0.90, 2, 6.0, "pluck"),
    "bass":       Timbre(0.008, 0.80, 0.45, 0.14, 1.15, 1, 0.0, "soft"),
    "strings":    Timbre(0.110, 0.35, 0.80, 0.40, 0.75, 3, 9.0, "saw"),
    "ensemble":   Timbre(0.150, 0.35, 0.85, 0.45, 0.70, 3, 11.0, "saw"),
    "brass":      Timbre(0.055, 0.22, 0.82, 0.18, 0.80, 2, 7.0, "brass"),
    "reed":       Timbre(0.040, 0.20, 0.85, 0.16, 0.78, 2, 5.0, "reed"),
    "pipe":       Timbre(0.050, 0.12, 0.92, 0.14, 0.72, 2, 4.0, "triangle"),
    "lead":       Timbre(0.010, 0.25, 0.75, 0.14, 0.78, 2, 12.0, "saw"),
    "pad":        Timbre(0.380, 0.60, 0.85, 0.90, 0.62, 4, 14.0, "soft"),
    "fx":         Timbre(0.250, 0.50, 0.60, 0.70, 0.60, 3, 16.0, "soft"),
    "ethnic":     Timbre(0.004, 0.85, 0.08, 0.22, 0.85, 2, 6.0, "pluck"),
    "percussive": Timbre(0.002, 0.35, 0.00, 0.20, 0.90, 1, 0.0, "bell"),
    "sfx":        Timbre(0.060, 0.40, 0.35, 0.35, 0.55, 2, 10.0, "soft"),
}
DEFAULT_TIMBRE = FAMILY_TIMBRE["piano"]


# --------------------------------------------------------------------------
# Wavetables
# --------------------------------------------------------------------------

class WaveTables:
    """Builds and caches the per-octave wavetables for each waveform.

    ``get("saw")`` returns a list of 11 arrays. Entry ``k`` is safe to play any
    MIDI note from ``k * 12`` to ``k * 12 + 11``: it holds only the harmonics
    that stay below :data:`MAX_FREQUENCY` for the highest note in that octave.
    High notes therefore get fewer harmonics, which is exactly what stops them
    aliasing.

    Each table is built once and kept for the life of the program.
    """

    _cache: dict[str, list[np.ndarray]] = {}

    @classmethod
    def get(cls, spectrum: str) -> list[np.ndarray]:
        """Return the list of octave tables for ``spectrum``, building it if needed."""
        if spectrum in cls._cache:
            return cls._cache[spectrum]

        tables = []
        # One full cycle, expressed as angles from 0 to 2*pi.
        angle = 2.0 * np.pi * np.arange(TABLE_SIZE) / TABLE_SIZE

        for octave in range(OCTAVES):
            highest = midi_to_freq(min(127, octave * 12 + 11))
            harmonic_limit = min(200, max(1, int(MAX_FREQUENCY / highest)))

            # Add up the harmonics that fit. This is an inverse Fourier
            # transform done the slow, obvious way -- fine, since it runs once.
            table = np.zeros(TABLE_SIZE, dtype=np.float64)
            for h in range(1, harmonic_limit + 1):
                amplitude = _harmonic_amplitude(spectrum, h)
                if amplitude:
                    table += amplitude * np.sin(h * angle)

            peak = np.max(np.abs(table))
            if peak > 1e-9:
                table /= peak       # normalise so every table is equally loud

            # Repeat the first sample at the end so interpolation between the
            # last and first sample needs no wrap-around check.
            tables.append(np.append(table, table[0]).astype(np.float32))

        cls._cache[spectrum] = tables
        return tables


def _oscillator(freq: float, n_samples: int, octave: int,
                tables: list[np.ndarray]) -> np.ndarray:
    """Play ``n_samples`` of a wavetable at ``freq`` Hz.

    Walks through the table at a constant step, wrapping around at the end, and
    blends between neighbouring samples so the pitch is accurate rather than
    quantised to whole table positions.

    Args:
        freq: the frequency to play, in Hz.
        n_samples: how many audio samples to produce.
        octave: which band-limited table to use, normally ``pitch // 12``.
            Using the table for the note's own octave is what prevents aliasing.
        tables: the list returned by :meth:`WaveTables.get`.
    """
    table = tables[min(max(octave, 0), OCTAVES - 1)]

    # How far to move through the table per output sample.
    step = freq * TABLE_SIZE / SAMPLE_RATE
    position = (np.arange(n_samples, dtype=np.float64) * step) % TABLE_SIZE

    index = position.astype(np.int32)
    fraction = (position - index).astype(np.float32)
    # Linear interpolation between the two table entries either side.
    return table[index] * (1.0 - fraction) + table[index + 1] * fraction


def _envelope(held_samples: int, timbre: Timbre, release_samples: int) -> np.ndarray:
    """Build the ADSR volume curve for one note.

    Returns:
        An array of length ``held_samples + release_samples``, starting at 0.0
        and ending at 0.0.
    """
    env = np.empty(held_samples + release_samples, dtype=np.float32)

    # Attack: a straight fade in.
    attack = min(held_samples, max(1, int(timbre.attack * SAMPLE_RATE)))
    env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)

    # Decay: fall from 1.0 towards the sustain level. Exponential curves sound
    # natural because that is how real instruments lose energy.
    if attack < held_samples:
        t = np.arange(held_samples - attack, dtype=np.float32) / SAMPLE_RATE
        env[attack:held_samples] = (
            timbre.sustain
            + (1.0 - timbre.sustain) * np.exp(-t / max(timbre.decay, 1e-4)))

    # Release: fade from wherever the note had got to down to silence. The
    # extra straight ramp guarantees it truly reaches zero, avoiding a click.
    if release_samples > 0:
        level = float(env[held_samples - 1])
        t = np.arange(release_samples, dtype=np.float32) / SAMPLE_RATE
        env[held_samples:] = (level
                              * np.exp(-t / max(timbre.release, 1e-4))
                              * np.linspace(1.0, 0.0, release_samples, dtype=np.float32))
    return env


class _NoteCache:
    """Remembers recently rendered notes so repeats are free.

    Music repeats constantly, so an arpeggio may play the same note hundreds of
    times. Keyed by instrument, pitch and rounded length, a note only has to be
    built once. Oldest entries are dropped when the cache grows past its budget
    (a "least recently used", or LRU, policy).
    """

    def __init__(self, max_samples: int = 40 * SAMPLE_RATE):
        """Args: max_samples: roughly how many seconds of audio to keep."""
        self._entries: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._max_samples = max_samples
        self._used = 0

    def get(self, key: tuple) -> np.ndarray | None:
        """Return the cached note for ``key``, or None. Marks it recently used."""
        buffer = self._entries.get(key)
        if buffer is not None:
            self._entries.move_to_end(key)
        return buffer

    def put(self, key: tuple, buffer: np.ndarray) -> None:
        """Store a note, evicting the least recently used ones if needed."""
        if buffer.size > self._max_samples // 4:
            return                       # one very long note must not fill the cache
        self._entries[key] = buffer
        self._used += buffer.size
        while self._used > self._max_samples and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._used -= evicted.size


def _render_pitched(family: str, pitch: int, duration: float,
                    cache: _NoteCache) -> np.ndarray:
    """Render one pitched (non-drum) note at full velocity.

    Note lengths are rounded to steps of a quarter-octave -- about 19% apart --
    before rendering, so that near-identical notes share one cache entry. The
    small timing difference is inaudible under an envelope that is fading anyway.

    Returns:
        A mono float32 array. The caller scales it for velocity and volume.
    """
    timbre = FAMILY_TIMBRE.get(family, DEFAULT_TIMBRE)

    length_step = round(math.log2(max(duration, 0.02)) * 4)
    key = (family, pitch, length_step)
    cached = cache.get(key)
    if cached is not None:
        return cached

    held = max(1, int(2.0 ** (length_step / 4.0) * SAMPLE_RATE))
    release = max(1, int(timbre.release * SAMPLE_RATE))

    tables = WaveTables.get(timbre.spectrum)
    base_freq = midi_to_freq(pitch)
    octave = pitch // 12
    out = np.zeros(held + release, dtype=np.float32)

    # Stack a few slightly detuned copies. Their small frequency differences
    # make them drift in and out of phase, which is what gives strings and pads
    # their shimmer instead of sounding like one thin tone.
    for voice in range(timbre.voices):
        offset = 0.0
        if timbre.voices > 1:
            spread = voice / (timbre.voices - 1) - 0.5     # -0.5 .. +0.5
            offset = timbre.detune * spread * 2.0          # in cents
        freq = base_freq * (2.0 ** (offset / 1200.0))
        if freq < MAX_FREQUENCY:
            out += _oscillator(freq, out.size, octave, tables)
    if timbre.voices > 1:
        out /= math.sqrt(timbre.voices)     # keep stacked voices from getting loud

    out *= _envelope(held, timbre, release) * timbre.gain
    cache.put(key, out)
    return out


# --------------------------------------------------------------------------
# Percussion
# --------------------------------------------------------------------------

def _decay_curve(duration: float, rate: float) -> np.ndarray:
    """A volume curve that starts at 1.0 and dies away. Higher ``rate`` = shorter."""
    n = max(1, int(duration * SAMPLE_RATE))
    env = np.exp(-rate * (np.arange(n, dtype=np.float32) / SAMPLE_RATE))
    fade = min(n, 256)
    env[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)   # avoid a click
    return env


@dataclass(frozen=True)
class Noise:
    """A burst of filtered noise -- the *sizzle* of a drum.

    Real noise is shaped here in the frequency domain: we give every frequency
    a chosen loudness and a random phase, then transform back to a waveform.
    That produces exactly the tone colour asked for in one step, with no filter
    to run sample by sample.
    """

    duration: float
    decay: float            # how fast it dies away
    tilt: float = 0.0       # above 0 favours high frequencies, brightening it
    lowcut: float = 0.0     # roll off everything below this many Hz
    level: float = 1.0      # loudness of this layer within the drum
    delay: float = 0.0      # start this many seconds into the drum

    def render(self, rng: np.random.Generator) -> np.ndarray:
        """Produce this layer's waveform."""
        n = max(16, int(self.duration * SAMPLE_RATE))
        bins = n // 2 + 1
        freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, bins)

        strength = np.ones(bins)
        if self.tilt:
            strength *= (np.maximum(freqs, 20.0) / 1000.0) ** self.tilt
        if self.lowcut > 0:
            strength *= 1.0 / (1.0 + (self.lowcut / np.maximum(freqs, 1.0)) ** 4)

        # A random phase per frequency is what makes it noise rather than a tone.
        spectrum = strength * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, bins))
        signal = np.fft.irfft(spectrum, n)

        peak = np.max(np.abs(signal))
        if peak > 1e-9:
            signal = signal / peak
        return (signal * _decay_curve(self.duration, self.decay)
                * self.level).astype(np.float32)


@dataclass(frozen=True)
class Body:
    """A sine whose pitch slides downwards -- the *thump* of a drum.

    Hitting a drum stretches its skin, so it starts sharp and settles to its
    natural pitch within a few hundredths of a second. Sweeping the frequency
    down reproduces that, and is most of what makes a kick sound like a kick.
    """

    start_freq: float
    end_freq: float
    duration: float
    decay: float
    sweep: float = 25.0     # how quickly the pitch settles
    level: float = 1.0
    delay: float = 0.0

    def render(self, rng: np.random.Generator) -> np.ndarray:
        """Produce this layer's waveform. ``rng`` is unused but keeps the API uniform."""
        n = max(1, int(self.duration * SAMPLE_RATE))
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        freq = self.end_freq + (self.start_freq - self.end_freq) * np.exp(-self.sweep * t)
        # Summing frequency over time gives phase, which is what a sine needs.
        phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        return (np.sin(phase) * _decay_curve(self.duration, self.decay)
                * self.level).astype(np.float32)


def _tom(base_freq: float) -> tuple:
    """Layers for a tom at the given pitch: a long body plus a short attack."""
    return (Body(base_freq * 1.7, base_freq, 0.42, decay=8.5, sweep=26.0),
            Noise(0.02, decay=160.0, tilt=0.3, lowcut=900, level=0.25))


def _hand_drum(base_freq: float) -> tuple:
    """Layers for a bongo or conga: like a tom, but tighter and struck by hand.

    The skin is smaller and under more tension than a tom's, so the note is
    shorter, settles to its pitch faster, and has a crisper slap on top.
    """
    return (Body(base_freq * 1.5, base_freq, 0.26, decay=15.0, sweep=34.0),
            Noise(0.015, decay=200.0, tilt=0.4, lowcut=1200, level=0.3))


def _cymbal(duration: float) -> tuple:
    """Layers for a crash or splash: broad noise plus a brighter shimmer on top."""
    return (Noise(duration, decay=3.2, tilt=0.55, lowcut=2200, level=0.50),
            Noise(duration, decay=5.0, tilt=0.95, lowcut=7000, level=0.22))


#: How each General MIDI drum is built, keyed by MIDI note number. Add a row to
#: teach the synth a new drum; anything missing falls back to _generic_drum().
DRUM_KIT: dict[int, tuple] = {
    35: (Body(165, 46, 0.42, decay=9.0, sweep=32.0, level=1.15),        # bass drum
         Noise(0.012, decay=260.0, tilt=0.4, lowcut=600, level=0.40)),
    37: (Noise(0.05, decay=80.0, tilt=0.5, lowcut=900),                 # side stick
         Body(950, 700, 0.05, decay=95.0, sweep=90.0, level=0.4)),
    38: (Noise(0.22, decay=21.0, tilt=0.25, lowcut=220, level=0.85),    # snare
         Body(330, 185, 0.22, decay=26.0, sweep=40.0, level=0.30),
         Body(460, 250, 0.22, decay=26.0, sweep=40.0, level=0.15)),
    # Hand clap: four quick bursts, because a clap is never a single sound.
    39: tuple(Noise(0.09, decay=42.0, tilt=0.35, lowcut=700,
                    level=1.0 - 0.16 * i, delay=0.011 * i) for i in range(4)),
    42: (Noise(0.055, decay=85.0, tilt=0.85, lowcut=5500, level=0.75),),  # closed hat
    44: (Noise(0.075, decay=85.0, tilt=0.85, lowcut=5500, level=0.75),),  # pedal hat
    46: (Noise(0.42, decay=9.5, tilt=0.8, lowcut=5000, level=0.7),),      # open hat
    51: (Noise(0.9, decay=6.0, tilt=0.7, lowcut=4200, level=0.45),        # ride
         Body(1180, 1150, 0.9, decay=7.5, sweep=3.0, level=0.35),
         Body(2360, 2300, 0.9, decay=7.5, sweep=3.0, level=0.175)),
    54: (Noise(0.28, decay=16.0, tilt=1.0, lowcut=6000, level=0.6),),     # tambourine
    56: (Body(835, 835, 0.32, decay=13.0, sweep=1.0, level=0.36),         # cowbell
         Body(560, 560, 0.32, decay=13.0, sweep=1.0, level=0.30)),
}

# Drums that share a recipe, filled in programmatically to keep the table short.
DRUM_KIT[36] = DRUM_KIT[35]                                  # the two bass drums
DRUM_KIT[40] = DRUM_KIT[38]                                  # electric snare
DRUM_KIT.update({53: DRUM_KIT[51], 59: DRUM_KIT[51]})        # ride bell, ride 2
DRUM_KIT.update({note: _tom(freq) for note, freq in
                 ((41, 92), (43, 110), (45, 132), (47, 158), (48, 190), (50, 226))})
DRUM_KIT.update({note: _hand_drum(freq) for note, freq in    # bongos and congas
                 ((60, 310), (61, 240), (62, 200), (63, 215), (64, 165))})
DRUM_KIT.update({49: _cymbal(1.6), 57: _cymbal(1.6),         # crashes ring longest
                 52: _cymbal(1.0), 55: _cymbal(1.0)})        # china, splash
DRUM_KIT.update({note: (Body(freq, freq * 0.95, 0.09, decay=55.0,   # claves, blocks
                             sweep=20.0, level=0.7),)
                 for note, freq in ((75, 2400), (76, 1200), (77, 900))})
DRUM_KIT.update({note: (Noise(0.1, decay=48.0, tilt=1.0,     # shakers and guiros
                              lowcut=5000, level=0.5),)
                 for note in (69, 70, 73, 74)})


def _generic_drum(note: int) -> tuple:
    """A plausible stand-in for any percussion note not listed in :data:`DRUM_KIT`."""
    freq = midi_to_freq(note) * 2.0
    return (Body(freq, freq * 0.8, 0.18, decay=22.0, sweep=30.0, level=0.5),
            Noise(0.03, decay=90.0, tilt=0.5, lowcut=1500, level=0.35))


#: Finished drum sounds, built on first use. Drums repeat far more than any
#: other note, so caching them matters most.
_DRUM_CACHE: dict[int, np.ndarray] = {}


def _render_drum(note: int) -> np.ndarray:
    """Render one drum at full velocity by mixing its layers together."""
    if note in _DRUM_CACHE:
        return _DRUM_CACHE[note]

    layers = DRUM_KIT.get(note) or _generic_drum(note)
    # Seeding from the note number keeps every render of the file identical,
    # while still giving each drum its own noise.
    rng = np.random.default_rng(note * 1013 + 7)

    total = max(int((layer.delay + layer.duration) * SAMPLE_RATE) for layer in layers)
    out = np.zeros(total, dtype=np.float32)
    for layer in layers:
        buffer = layer.render(rng)
        start = int(layer.delay * SAMPLE_RATE)
        end = min(total, start + buffer.size)
        out[start:end] += buffer[:end - start]

    _DRUM_CACHE[note] = out
    return out


# --------------------------------------------------------------------------
# Mixing and effects
# --------------------------------------------------------------------------

def _reverb_impulse(seconds: float = 1.05) -> np.ndarray:
    """Build a fake room echo: noise that fades out, with the highs damped.

    Playing a sound "through" this (see :func:`_convolve`) makes it sound as if
    it happened in a room, because that is literally what a room does to sound.

    Returns:
        A ``(2, n)`` array -- one impulse per stereo channel, slightly different
        so the result sounds wide rather than flat.
    """
    rng = np.random.default_rng(1234)
    n = int(seconds * SAMPLE_RATE)
    fade = np.exp(-4.2 * np.arange(n) / SAMPLE_RATE)

    impulse = np.empty((2, n), dtype=np.float32)
    for channel in range(2):
        spectrum = np.fft.rfft(rng.standard_normal(n) * fade)
        freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, spectrum.size)
        spectrum *= 1.0 / (1.0 + (freqs / 3200.0) ** 1.6)   # soft surfaces eat highs
        signal = np.fft.irfft(spectrum, n)
        signal[:int(0.012 * SAMPLE_RATE)] = 0.0             # gap before the first echo
        impulse[channel] = signal / (np.max(np.abs(signal)) + 1e-9)
    return impulse


def _convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """Apply an impulse response to a signal, in blocks.

    Convolution is slow done directly, but multiplication in the frequency
    domain is equivalent and fast. Working a block at a time keeps memory use
    flat; each block's tail overlaps into the next, which is why this technique
    is called *overlap-add*.

    Returns:
        An array the same length as ``signal``.
    """
    impulse_len = impulse.size
    # Block size: a power of two comfortably larger than the impulse.
    block = 1 << max(12, int(math.ceil(math.log2(max(impulse_len * 4, 4096)))))
    step = block - impulse_len + 1
    if step <= 0:
        return signal

    impulse_spectrum = np.fft.rfft(impulse, block)
    out = np.zeros(signal.size + impulse_len, dtype=np.float32)
    for start in range(0, signal.size, step):
        chunk = signal[start:start + step]
        if chunk.size == 0:
            break
        block_out = np.fft.irfft(np.fft.rfft(chunk, block) * impulse_spectrum, block)
        end = min(start + block_out.size, out.size)
        out[start:end] += block_out[:end - start].astype(np.float32)
    return out[:signal.size]


def _stereo_gains(pan: float, amplitude: float) -> tuple[float, float]:
    """Split one amplitude into left and right according to ``pan``.

    Args:
        pan: -1.0 hard left, 0.0 centre, +1.0 hard right.

    Uses a quarter-circle curve rather than a straight split so the perceived
    loudness stays constant as a sound moves across the stereo field.
    """
    angle = (max(-1.0, min(1.0, pan)) + 1.0) * 0.25 * math.pi
    return math.cos(angle) * amplitude, math.sin(angle) * amplitude


def _master(stereo: np.ndarray) -> np.ndarray:
    """Final polish: level the mix, tame peaks, then normalise.

    ``tanh`` squashes the loudest peaks smoothly instead of chopping their tops
    off, which is the difference between a warm mix and an audibly broken one.
    """
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1e-6:
        stereo = stereo * min(3.0, 0.82 / peak)

    stereo = np.tanh(stereo * 1.12) * 0.93

    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1e-6:
        stereo = stereo * (0.89 / peak)     # leave a little headroom below 1.0
    return stereo.astype(np.float32)


def render_score(score: Score, progress=None, cancel=None,
                 reverb: float = 0.22) -> np.ndarray:
    """Render a whole :class:`~midiviz.midi_parser.Score` to stereo audio.

    Args:
        score: the parsed MIDI.
        progress: optional callback taking a float from 0.0 to 1.0.
        cancel: optional callback returning True to abort early.
        reverb: how much room echo to add, 0.0 to 1.0.

    Returns:
        A float32 array shaped ``(samples, 2)``, values between -1.0 and 1.0.
    """
    # One extra second of room so the last note's reverb tail is not cut off.
    total_samples = int(math.ceil(score.duration * SAMPLE_RATE)) + SAMPLE_RATE
    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)

    cache = _NoteCache()
    notes = sorted(((note, track) for track in score.tracks for note in track.notes),
                   key=lambda pair: pair[0].start)
    track_count = max(1, len(score.tracks))
    # More tracks playing at once means more chance of overload, so pull
    # everything down a little as the arrangement gets denser.
    density_trim = 1.0 / math.sqrt(max(1.0, track_count * 0.55))

    for i, (note, track) in enumerate(notes):
        if cancel is not None and cancel():
            break
        if progress is not None and i % 128 == 0:
            # The note loop is most of the work but not all; leave the last
            # tenth for the reverb and mastering below.
            progress(0.90 * i / len(notes))

        # Velocity to loudness. The power curve widens the gap between soft and
        # hard notes, which matches how we hear and makes playing feel dynamic.
        amplitude = (note.velocity / 127.0) ** 1.55
        amplitude *= (0.30 + 0.70 * track.volume) * density_trim

        if track.is_drum:
            buffer = _render_drum(note.pitch)
            amplitude *= 0.95
        else:
            buffer = _render_pitched(track.family, note.pitch, note.duration, cache)
            # Low notes carry more energy than the ear needs, so tilt them down.
            amplitude *= 0.55 + 0.45 * min(1.0, (note.pitch + 12) / 90.0)

        start = int(note.start * SAMPLE_RATE)
        end = min(total_samples, start + buffer.size)
        if end <= start:
            continue

        # Place the track in the stereo field: use its own pan if the file set
        # one, otherwise fan the tracks out evenly so they do not pile up.
        pan = track.pan
        if abs(pan) < 1e-6 and track_count > 1:
            pan = (track.index / (track_count - 1) - 0.5) * 1.05
        gain_left, gain_right = _stereo_gains(pan, amplitude)

        segment = buffer[:end - start]
        left[start:end] += segment * gain_left
        right[start:end] += segment * gain_right

    if progress is not None:
        progress(0.92)

    if reverb > 0.001 and total_samples > SAMPLE_RATE // 2:
        impulse = _reverb_impulse()
        # Mix the echoed ("wet") sound back in under the original ("dry").
        left = left * (1.0 - reverb * 0.45) + _convolve(left, impulse[0]) * reverb * 0.5
        right = right * (1.0 - reverb * 0.45) + _convolve(right, impulse[1]) * reverb * 0.5

    if progress is not None:
        progress(1.0)
    return _master(np.stack([left, right], axis=1))


def write_wav(path: str, audio: np.ndarray) -> str:
    """Save a float ``(samples, 2)`` array as a 16-bit stereo WAV file.

    Returns:
        ``path``, for convenience when chaining calls.
    """
    samples = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)          # 2 bytes per sample = 16 bit
        out.setframerate(SAMPLE_RATE)
        out.writeframes(samples.tobytes())
    return path
