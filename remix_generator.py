import librosa
import soundfile as sf
import os

REMIX_FOLDER = "../remixes"

os.makedirs(
    REMIX_FOLDER,
    exist_ok=True
)


def generate_slow_remix(filepath):

    y, sr = librosa.load(
        filepath,
        sr=None
    )

    slow_audio = librosa.effects.time_stretch(
        y=y,
        rate=0.8
    )

    output_file = os.path.join(
        REMIX_FOLDER,
        "slow_remix.wav"
    )

    sf.write(
        output_file,
        slow_audio,
        sr
    )

    return output_file


def generate_fast_remix(filepath):

    y, sr = librosa.load(
        filepath,
        sr=None
    )

    fast_audio = librosa.effects.time_stretch(
        y=y,
        rate=1.2
    )

    output_file = os.path.join(
        REMIX_FOLDER,
        "fast_remix.wav"
    )

    sf.write(
        output_file,
        fast_audio,
        sr
    )

    return output_file


def generate_pitch_remix(filepath):

    y, sr = librosa.load(
        filepath,
        sr=None
    )

    pitch_audio = librosa.effects.pitch_shift(
        y=y,
        sr=sr,
        n_steps=4
    )

    output_file = os.path.join(
        REMIX_FOLDER,
        "pitch_remix.wav"
    )

    sf.write(
        output_file,
        pitch_audio,
        sr
    )

    return output_file


def generate_lofi_remix(filepath):

    y, sr = librosa.load(
        filepath,
        sr=None
    )

    # Slow audio slightly
    y = librosa.effects.time_stretch(
        y=y,
        rate=0.9
    )

    # Lower pitch
    y = librosa.effects.pitch_shift(
        y=y,
        sr=sr,
        n_steps=-2
    )

    output_file = os.path.join(
        REMIX_FOLDER,
        "lofi_remix.wav"
    )

    sf.write(
        output_file,
        y,
        sr
    )

    return output_file