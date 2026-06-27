import librosa
import numpy as np

def extract_audio_features(audio_path):
    audio, sr = librosa.load(audio_path)

    duration = librosa.get_duration(y=audio, sr=sr)
    rms_energy = float(np.mean(librosa.feature.rms(y=audio)))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

    return {
        "duration": round(duration, 2),
        "rms_energy": round(rms_energy, 4),
        "zero_crossing_rate": round(zero_crossing_rate, 4)
    }

def filler_word_ratio(transcript):
    filler_words = ["um", "uh", "like", "you know", "actually", "basically"]
    words = transcript.lower().split()

    if len(words) == 0:
        return 0

    filler_count = sum(words.count(word) for word in filler_words)
    return round((filler_count / len(words)) * 100, 2)