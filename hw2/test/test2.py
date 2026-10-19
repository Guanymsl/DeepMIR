import subprocess
import numpy as np
import pretty_midi
import soundfile as sf

from pathlib import Path

from tokenization_corrected import corrected_score

MIDI_PATH = Path("data/POP909/prompts/basic-b/003/003.mid")
BEAT_PATH = Path("data/POP909/prompts/basic-b/003/beat_midi.txt")
SF2_PATH = Path("soundfonts/FluidR3_GM.sf2")

OUTPUT_DIR = Path("outputs/test_corrected")
SAMPLE_RATE = 44100

def midi_to_wav(midi_path, wav_path):
    subprocess.run(
        [
            "fluidsynth",
            "-ni",
            "-F", str(wav_path),
            "-r", str(SAMPLE_RATE),
            str(SF2_PATH),
            str(midi_path),
        ],
        check=True,
    )

def get_notes(midi):
    notes = []

    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append((
                note.pitch,
                note.velocity,
                note.start,
                note.end,
            ))

    return sorted(notes)

def compare_midi(original_path, corrected_path):
    original = pretty_midi.PrettyMIDI(str(original_path))
    corrected = pretty_midi.PrettyMIDI(str(corrected_path))

    a = get_notes(original)
    b = get_notes(corrected)

    print("\n===== MIDI Notes =====")
    print("Original notes: ", len(a))
    print("Corrected notes:", len(b))

    if len(a) != len(b):
        print("WARNING: Note counts differ!")
        return

    pitch_equal = all(x[0] == y[0] for x, y in zip(a, b))
    velocity_equal = all(x[1] == y[1] for x, y in zip(a, b))

    start_errors = np.array([
        abs(x[2] - y[2]) for x, y in zip(a, b)
    ])
    end_errors = np.array([
        abs(x[3] - y[3]) for x, y in zip(a, b)
    ])

    print("Pitch identical:   ", pitch_equal)
    print("Velocity identical:", velocity_equal)

    if len(a) > 0:
        print(f"Mean start error: {np.mean(start_errors) * 1000:.4f} ms")
        print(f"Max start error:  {np.max(start_errors) * 1000:.4f} ms")
        print(f"Mean end error:   {np.mean(end_errors) * 1000:.4f} ms")
        print(f"Max end error:    {np.max(end_errors) * 1000:.4f} ms")

    print("\n===== Tempo =====")

    original_times, original_bpms = original.get_tempo_changes()
    corrected_times, corrected_bpms = corrected.get_tempo_changes()

    print("Original tempo events:", len(original_bpms))
    print("Corrected tempo events:", len(corrected_bpms))

    print("Original first tempos:")
    for t, bpm in list(zip(original_times, original_bpms))[:5]:
        print(f"  {t:.6f} s -> {bpm:.4f} BPM")

    print("Corrected first tempos:")
    for t, bpm in list(zip(corrected_times, corrected_bpms))[:5]:
        print(f"  {t:.6f} s -> {bpm:.4f} BPM")

    print("\n===== Time Signatures =====")

    print("Original:")
    for ts in original.time_signature_changes:
        print(f"  {ts.numerator}/{ts.denominator} at {ts.time:.6f} s")

    print("Corrected:")
    for ts in corrected.time_signature_changes:
        print(f"  {ts.numerator}/{ts.denominator} at {ts.time:.6f} s")


def compare_wav(original_path, corrected_path):
    original, sr1 = sf.read(original_path)
    corrected, sr2 = sf.read(corrected_path)

    if sr1 != sr2:
        raise ValueError("Sample rates do not match")

    if original.shape[1:] != corrected.shape[1:]:
        raise ValueError("Audio channel counts do not match")

    n = min(len(original), len(corrected))

    a = original[:n]
    b = corrected[:n]
    diff = a - b

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    max_diff = np.max(np.abs(diff))

    print("\n===== WAV =====")
    print(f"Sample rate:         {sr1} Hz")
    print(f"Original duration:   {len(original) / sr1:.6f} s")
    print(f"Corrected duration:  {len(corrected) / sr2:.6f} s")
    print(f"Duration difference: {abs(len(original) - len(corrected)) / sr1:.6f} s")

    print("\n===== Audio Difference =====")
    print(f"RMSE:     {rmse:.8f}")
    print(f"MAE:      {mae:.8f}")
    print(f"Max diff: {max_diff:.8f}")

def main():
    for path in [MIDI_PATH, BEAT_PATH, SF2_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corrected_midi = OUTPUT_DIR / "corrected.mid"
    original_wav = OUTPUT_DIR / "original.wav"
    corrected_wav = OUTPUT_DIR / "corrected.wav"

    score, changes = corrected_score(MIDI_PATH, BEAT_PATH)
    score.dump_midi(corrected_midi)

    print("===== Correction =====")
    print("Time signature changes:", changes)

    compare_midi(MIDI_PATH, corrected_midi)

    print("\n===== Rendering WAV =====")
    midi_to_wav(MIDI_PATH, original_wav)
    midi_to_wav(corrected_midi, corrected_wav)

    compare_wav(original_wav, corrected_wav)

    print("\n===== Files =====")
    print("Corrected MIDI:", corrected_midi)
    print("Original WAV:  ", original_wav)
    print("Corrected WAV: ", corrected_wav)

if __name__ == "__main__":
    main()
