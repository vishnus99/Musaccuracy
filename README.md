# Musaccuracy

An app that allows you to compare your own music recording to the original and recieve an accuracy grade based on pitch accuracy and rhythmic similarity. 


## Audio Processing
The app uses the Demucs library to separate the given mp3 files into the stems, or the individual instruments and vocals. Using the librosa library, the app then compares the chromagrams of the cover's audio files and the original song's audio file. The app can handle different audio lengths and tempos, as well as .mp3 and .wav files. 

## Musical Analysis
60% of the accuracy score is based on the pitch comparison between the cover and the original song. This comparison is done using chromagram analysis. Chromagrams are a way of representing pitch class content in which all octaves are flattened into one of the 12 pitch classes. This allows for a more forgiving comparison between the cover and the original song since the octave is not a factor. The librosa library is used to generate the chromagrams. Finally, Dynamic Time Warping is used to provide even more flexibility in grading. DTW is an algorithm that finds the optimal alignment between two sequences, in this case, it allows for the cover's chromagram to be compared to the original's despite tempo differences.

The remaining 40% of the accuracy score is based on the rhythmic comparison between the cover and original song. This is done using onset detection. Onsets are the times at which notes or percussive events begin in an audio signal. In a given signal, we can detect the onsets as well as their strength. For the rhythmic comparison, the timing, strength, and pattern of the onsets are compared between the cover and the original song. 

## Scoring System
The app uses a weighted scoring system to combine the pitch and rhythmic accuracy scores:
- 1.0 to 0.8: Excellent match
- 0.8 to 0.6: Good match
- 0.6 to 0.4: Moderate match
- 0.4 to 0.2: Poor match
- 0.2 to 0.0: Very poor match

## Techinal Components
- Demucs: https://github.com/facebookresearch/demucs
- Librosa: https://github.com/librosa/librosa
- numpy: https://numpy.org/
- matplotlib: https://matplotlib.org/

## Additional Resources
- Dynamic Time Warping: https://en.wikipedia.org/wiki/Dynamic_time_warping
- Onset Detection: https://en.wikipedia.org/wiki/Onset_detection
- Chroma: https://en.wikipedia.org/wiki/Chroma_feature
- Chromagram: https://en.wikipedia.org/wiki/Chromagram

## Example Chromagram Comparison
![Figure_1](https://github.com/user-attachments/assets/9835daf3-6e04-4523-88e5-2be4adbda1ef)





