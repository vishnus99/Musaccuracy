import librosa
import subprocess
import demucs
from helper_functions import *

mp3_file = "mozartrequiem.mp3"
cover_file = "mozartcover.mp3"
#split_audio_file(mp3_file)
original_audio, cover_audio = load_in_mp3(mp3_file, "other", cover_file)