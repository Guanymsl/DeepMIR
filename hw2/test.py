import pretty_midi

midi = pretty_midi.PrettyMIDI(
    "data/POP909/prompts/basic-b/001/001.mid"
)

times, tempos = midi.get_tempo_changes()

print("Tempo changes:", list(zip(times, tempos)))
print("MIDI end time:", midi.get_end_time())