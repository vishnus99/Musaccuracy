import librosa
import demucs
import subprocess
import numpy as np
import librosa.display
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


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
    original_audio_file_path = f"./separated/htdemucs/{audio_filename}/{instrument}.wav"
    cover_audio_file_path = f"./My Audio Files/{your_mp3}"
    y_original, sr_original = librosa.load(original_audio_file_path)
    y_cover, sr_cover  = librosa.load(cover_audio_file_path)
    reshape_mfccs(y_original, sr_original, y_cover, sr_cover)
    original_audio_object = [y_original, sr_original]
    cover_audio_object = [y_cover, sr_cover]

    return original_audio_object, cover_audio_object

def reshape_mfccs(y_original, sr_original, y_cover, sr_cover):
    # Compute MFCCs for both
    mfccs1 = librosa.feature.mfcc(y=y_original, sr=sr_original)
    mfccs2 = librosa.feature.mfcc(y=y_cover, sr=sr_cover)

    # Ensure the shapes are correct
    print(mfccs1.shape, mfccs2.shape)  # Should both be (n_mfcc, time_frames)
    # Get the number of time frames for both matrices
    time_frames_1 = mfccs1.shape[1]
    time_frames_2 = mfccs2.shape[1]

    # Determine which one is shorter
    if time_frames_1 < time_frames_2:
    # Pad mfccs1 to match mfccs2
        pad_width = time_frames_2 - time_frames_1
        mfccs1 = np.pad(mfccs1, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Pad mfccs2 to match mfccs1
        pad_width = time_frames_1 - time_frames_2
        mfccs2 = np.pad(mfccs2, ((0, 0), (0, pad_width)), mode='constant')

    # Check the shapes again to confirm they're equal
    return(mfccs1, mfccs2)

def extract_musical_features(original_obj, cover_obj):
    # Extract chroma features (harmonic content) from both recordings
    chroma_original = librosa.feature.chroma_cqt(y=original_obj[0], sr=original_obj[1])
    chroma_cover = librosa.feature.chroma_cqt(y=cover_obj[0], sr=cover_obj[1])

    # Check the number of time frames for both chroma matrices
    time_frames_original = chroma_original.shape[1]
    time_frames_cover = chroma_cover.shape[1]

    # Pad the shorter one
    if time_frames_original < time_frames_cover:
        pad_width = time_frames_cover - time_frames_original
        chroma_original = np.pad(chroma_original, ((0, 0), (0, pad_width)), mode='constant')
    elif time_frames_cover < time_frames_original:
        pad_width = time_frames_original - time_frames_cover
        chroma_cover = np.pad(chroma_cover, ((0, 0), (0, pad_width)), mode='constant')

    chroma_obj = [chroma_original, chroma_cover]

    # Extract onset envelopes (representing rhythmic beats) from both recordings
    onset_env_original = librosa.onset.onset_strength(y=original_obj[0], sr=original_obj[1])
    onset_env_cover = librosa.onset.onset_strength(y=cover_obj[0], sr=cover_obj[1])
    onset_obj = [onset_env_original, onset_env_cover]

    return chroma_obj, onset_obj


def plot_chroma_accuracy(chroma_obj):
    # Compute the distance between chroma features
    D, wp = librosa.sequence.dtw(X=chroma_obj[0].T, Y=chroma_obj[1].T, metric='cosine')

    # Plot the results
    plt.figure(figsize=(8, 8))
    librosa.display.specshow(D, x_axis='time', y_axis='time')
    plt.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='r')
    plt.title('Dynamic Time Warping (DTW) between original and recorded chroma features')
    plt.legend()
    plt.show()
    calculate_chroma_accuracy(D)


def calculate_chroma_accuracy(D):
    # Calculate the DTW distance
    dtw_distance = np.sum(D)
    accuracy = 1 / (1 + dtw_distance)  # Simple scaling of the distance to get an accuracy score between 0 and 1

    print(f"Accuracy between original and cover performance: {accuracy:.2f}")