"""General MIDI instrument names and families.

A MIDI file does not contain any sound, only a *program number* from 0 to 127
saying which instrument to use. The General MIDI standard fixes what each number
means, so 0 is always a grand piano and 40 always a violin.

This module turns those numbers into names for the labels in the video, and into
family names that tell the synthesizer roughly what the instrument should sound
like.
"""

#: The 128 General MIDI instrument names, in order. Index = program number.
GM_PROGRAMS = [
    "Acoustic Grand Piano", "Bright Acoustic Piano",
    "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
    "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Choir", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass+lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto",
    "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]

#: General MIDI groups its instruments into sixteen blocks of eight, and each
#: block holds instruments that sound broadly alike. One name per block, so the
#: synthesizer needs sixteen tones rather than a hundred and twenty-eight.
FAMILIES = [
    "piano", "chromatic", "organ", "guitar", "bass", "strings", "ensemble", "brass",
    "reed", "pipe", "lead", "pad", "fx", "ethnic", "percussive", "sfx",
]


def program_name(program: int, is_drum: bool = False) -> str:
    """Return the instrument name for a General MIDI program number.

    Args:
        program: the program number; values outside 0-127 are clamped.
        is_drum: True for the percussion channel, where the program number
            does not select an instrument at all.
    """
    if is_drum:
        return "Drum Kit"
    return GM_PROGRAMS[max(0, min(127, int(program)))]


#: A few instruments sit in a block of eight that misrepresents them, because
#: General MIDI had to put them somewhere. Pizzicato strings and the harp are
#: plucked rather than bowed, and a timpani is a drum -- yet all three are
#: filed under "strings". Overriding them makes the synth sound much closer.
_FAMILY_OVERRIDES = {
    45: "ethnic",       # Pizzicato Strings -- plucked, short decay
    46: "ethnic",       # Orchestral Harp -- likewise
    47: "percussive",   # Timpani -- a struck drum
}


def family_of(program: int, is_drum: bool = False) -> str:
    """Return the instrument family for a program number, e.g. ``"bass"``.

    The synthesizer looks the result up in ``synth.FAMILY_TIMBRE`` to decide how
    the instrument should sound.
    """
    if is_drum:
        return "drums"
    program = max(0, min(127, int(program)))
    if program in _FAMILY_OVERRIDES:
        return _FAMILY_OVERRIDES[program]
    return FAMILIES[program // 8]
