"""MIDI Visualizer: turn a MIDI file into a video with animated bars.

Modules, in the order the data flows through them:

* :mod:`midiviz.midi_parser` -- read a MIDI file into a plain ``Score``
* :mod:`midiviz.synth`       -- turn the score into audio
* :mod:`midiviz.renderer`    -- turn the score into animation frames
* :mod:`midiviz.video`       -- feed frames and audio to ffmpeg
* :mod:`midiviz.pipeline`    -- run all of the above in order
* :mod:`midiviz.ui`          -- the PyQt6 window
* :mod:`midiviz.gm`          -- General MIDI instrument names and families
"""

__version__ = "1.0.0"
