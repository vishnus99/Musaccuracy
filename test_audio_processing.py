import pytest
import numpy as np
import librosa
from unittest import mock
from main import split_audio_file, load_in_mp3, reshape_mfccs, extract_musical_features, plot_chroma_accuracy, calculate_chroma_accuracy

# Test split_audio_file function
@mock.patch("subprocess.run")
def test_split_audio_file(mock_subprocess):
    # Mock subprocess run output
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = "Mocked: Separation completed successfully!"
    
    split_audio_file("test_audio.mp3")
    mock_subprocess.assert_called_once()

# Test load_in_mp3 function
@mock.patch("librosa.load")
def test_load_in_mp3(mock_load):
    # Mock loading of audio files
    mock_load.return_value = (np.random.rand(22050), 22050)
    
    original_audio, cover_audio = load_in_mp3("test_audio.mp3", "vocals", "cover_audio.mp3")
    assert original_audio[0].shape == (22050,)
    assert cover_audio[0].shape == (22050,)

# Test reshape_mfccs function
def test_reshape_mfccs():
    # Generate sample data
    y_original = np.random.rand(22050)
    sr_original = 22050
    y_cover = np.random.rand(21000)
    sr_cover = 22050
    
    mfccs1, mfccs2 = reshape_mfccs(y_original, sr_original, y_cover, sr_cover)
    
    # Check if shapes match after padding
    assert mfccs1.shape == mfccs2.shape

# Test extract_musical_features function
@mock.patch("librosa.feature.chroma_cqt")
@mock.patch("librosa.onset.onset_strength")
def test_extract_musical_features(mock_onset, mock_chroma):
    # Mock chroma and onset envelope extraction
    mock_chroma.return_value = np.random.rand(12, 100)
    mock_onset.return_value = np.random.rand(100)
    
    original_obj = [np.random.rand(22050), 22050]
    cover_obj = [np.random.rand(22050), 22050]
    
    chroma_obj, onset_obj = extract_musical_features(original_obj, cover_obj)
    
    # Verify the output shapes
    assert chroma_obj[0].shape == chroma_obj[1].shape
    assert onset_obj[0].shape == onset_obj[1].shape

# Test plot_chroma_accuracy function
@mock.patch("librosa.sequence.dtw")
@mock.patch("librosa.display.specshow")
def test_plot_chroma_accuracy(mock_specshow, mock_dtw):
    # Mock DTW calculation and plotting
    mock_dtw.return_value = (np.random.rand(10, 10), np.array([[0, 1], [2, 3]]))
    chroma_obj = [np.random.rand(12, 100), np.random.rand(12, 100)]
    
    plot_chroma_accuracy(chroma_obj)
    mock_specshow.assert_called_once()

# Test calculate_chroma_accuracy function
def test_calculate_chroma_accuracy():
    # Mock distance matrix
    D = np.random.rand(10, 10)
    
    # Capture print output
    with mock.patch("builtins.print") as mock_print:
        calculate_chroma_accuracy(D)
        mock_print.assert_called_once()

