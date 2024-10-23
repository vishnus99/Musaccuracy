import librosa
import demucs
import subprocess

#Demucs shell command
def split_audio_file(audio_file):
    command = ("demucs", "-v", f"./Audio Files/{audio_file}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print("Separation completed successfully!")
        print(result.stdout)
    else:
        print(f"An error occurred: {result.stderr}")

#Load in the recording to compare to and the submitted recording
def load_in_mp3(audio_file, instrument, your_mp3):
    audio_filename = audio_file.rsplit(".", -1)[0]
    y_original, sr_original = librosa.load(f"./separated/htdemucs/{audio_filename}/{instrument}.wav")
    y_cover, sr_cover  = librosa.load(f"./My Audio Files/{your_mp3}")
    original_audio_object = [y_original, sr_original]
    cover_audio_object = [y_cover, sr_cover]
    return original_audio_object, cover_audio_object

def extract_musical_features(original_obj, cover_obj):
    # Extract chroma features (harmonic content) from both recordings
    chroma_original = librosa.feature.chroma_cqt(y=original_obj[0], sr=original_obj[1])
    chroma_cover = librosa.feature.chroma_cqt(y=cover_obj[0], sr=cover_obj[1])
    chroma_obj = [chroma_original, chroma_cover]
    # Extract onset envelopes (representing rhythmic beats) from both recordings
    onset_env_original = librosa.onset.onset_strength(y=original_obj[0], sr=original_obj[1])
    onset_env_cover = librosa.onset.onset_strength(y=cover_obj[0], sr=cover_obj[1])
    onset_obj = [onset_env_original, onset_env_cover]
    
    return chroma_obj, onset_obj
