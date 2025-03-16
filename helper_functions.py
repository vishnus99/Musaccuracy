import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import subprocess
from typing import Tuple, List, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, hop_length: int = 1024, max_length: int = 500):
        """Initialize audio processing parameters."""
        self.hop_length = hop_length
        self.max_length = max_length  # Maximum sequence length for DTW
    
    def split_audio_file(self, original_audio_file: str, cover_audio_file: str) -> None:
        """Split audio file using Demucs."""
        try:
            # Check if file exists first
            original_audio_path = os.path.join("Audio Files", original_audio_file)
            cover_audio_path = os.path.join("My Audio Files", cover_audio_file)
            if not os.path.exists(original_audio_path):
                raise FileNotFoundError(f"Original audio file not found: {original_audio_path}")
            if not os.path.exists(cover_audio_path):
                raise FileNotFoundError(f"Cover audio file not found: {cover_audio_path}")
            
            # Use absolute path to avoid any path issues
            abs_original_path = os.path.abspath(original_audio_path)
            abs_cover_path = os.path.abspath(cover_audio_path)
            logger.info(f"Processing file: {abs_original_path}")
            logger.info(f"Processing file: {abs_cover_path}")
            
            original_audio_command = ["demucs", "-v", abs_original_path]
            cover_audio_command = ["demucs", "-v", abs_cover_path]
            original_result = subprocess.run(original_audio_command, capture_output=True, text=True)
            cover_result = subprocess.run(cover_audio_command, capture_output=True, text=True)
            
            if original_result.returncode == 0:
                logger.info("Separation completed successfully!")
                logger.debug(original_result.stdout)
            else:
                logger.error(f"Separation failed: {original_result.stderr}")
                logger.error(f"Command used: {' '.join(original_audio_command)}")
            if cover_result.returncode == 0:
                logger.info("Separation completed successfully!")
                logger.debug(cover_result.stdout)
            else:
                logger.error(f"Separation failed: {cover_result.stderr}")
                logger.error(f"Command used: {' '.join(cover_audio_command)}")
        except Exception as e:
            logger.error(f"Error during audio separation: {str(e)}")
            raise

    def load_audio_files(self, audio_file: str, instrument: str, your_mp3: str) -> Tuple[List, List]:
        """Load and prepare audio files for comparison."""
        try:
            # Prepare file paths
            original_audio_filename = audio_file.rsplit(".", -1)[0]
            cover_audio_filename = your_mp3.rsplit(".", -1)[0]
            original_path = f"./separated/htdemucs/{original_audio_filename}/{instrument}.wav"
            cover_path = f"./separated/htdemucs/{cover_audio_filename}/{instrument}.wav"
            
            # Load audio files
            y_original, sr_original = librosa.load(original_path)
            y_cover, sr_cover = librosa.load(cover_path)
            
            logger.info(f"Loaded files - Original SR: {sr_original}, Cover SR: {sr_cover}")
            return [y_original, sr_original], [y_cover, sr_cover]
            
        except Exception as e:
            logger.error(f"Error loading audio files: {str(e)}")
            raise

    def extract_features(self, original_obj: List, cover_obj: List) -> Tuple[List, List]:
        """Extract musical features from audio files."""
        try:
            # Normalize audio signals
            original_audio = librosa.util.normalize(original_obj[0])
            cover_audio = librosa.util.normalize(cover_obj[0])
            
            # Extract chroma features
            chroma_original = librosa.feature.chroma_cqt(
                y=original_audio, 
                sr=original_obj[1],
                norm=2,
                hop_length=self.hop_length
            )
            
            chroma_cover = librosa.feature.chroma_cqt(
                y=cover_audio, 
                sr=cover_obj[1],
                norm=2,
                hop_length=self.hop_length
            )
            
            # Find best matching segment
            segment_length = chroma_cover.shape[1]
            best_start = self._find_best_segment(chroma_original, chroma_cover)
            
            # Extract and normalize segments
            chroma_original_segment = chroma_original[:, best_start:best_start + segment_length]
            chroma_original_segment = librosa.util.normalize(chroma_original_segment, axis=0)
            chroma_cover = librosa.util.normalize(chroma_cover, axis=0)
            
            # Extract onset envelopes
            onset_env_original = librosa.onset.onset_strength(
                y=original_audio[best_start * self.hop_length:
                               (best_start + segment_length) * self.hop_length],
                sr=original_obj[1]
            )
            onset_env_cover = librosa.onset.onset_strength(
                y=cover_audio, 
                sr=cover_obj[1]
            )
            
            return [chroma_original_segment, chroma_cover], [onset_env_original, onset_env_cover]
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def _find_best_segment(self, chroma_original: np.ndarray, chroma_cover: np.ndarray) -> int:
        """Find the best matching segment in the original audio."""
        segment_length = chroma_cover.shape[1]
        best_start = 0
        min_distance = float('inf')
        
        # Use vectorized operations for faster computation
        for start in range(0, chroma_original.shape[1] - segment_length, self.hop_length):
            segment = chroma_original[:, start:start + segment_length]
            dist = np.mean(np.abs(segment - chroma_cover))
            if dist < min_distance:
                min_distance = dist
                best_start = start
        
        logger.info(f"Best matching segment found at frame {best_start}")
        return best_start

    def compare_rhythm(self, onset_obj: List) -> float:
        """Compare rhythmic similarity using onset patterns."""
        try:
            # Ensure onsets are not empty and contain valid values
            onset_orig = np.array(onset_obj[0]).reshape(1, -1)
            onset_cover = np.array(onset_obj[1]).reshape(1, -1)
            
            # Check for invalid values
            if np.any(np.isnan(onset_orig)) or np.any(np.isnan(onset_cover)):
                logger.warning("Found NaN values in onset data, replacing with zeros")
                onset_orig = np.nan_to_num(onset_orig)
                onset_cover = np.nan_to_num(onset_cover)
            
            # Ensure non-zero values for normalization
            if np.all(onset_orig == 0) or np.all(onset_cover == 0):
                logger.warning("Found zero onset strength, rhythm comparison may not be meaningful")
                return 0.5  # Return neutral score
            
            # Normalize onset envelopes
            onset_orig = librosa.util.normalize(onset_orig, axis=1)
            onset_cover = librosa.util.normalize(onset_cover, axis=1)
            
            # Compute onset similarity using DTW
            D_rhythm, _ = librosa.sequence.dtw(
                onset_orig, 
                onset_cover,
                metric='cosine'
            )
            
            # Calculate rhythm score using same percentile method
            p10, p25, p50, p75, p90 = np.percentile(D_rhythm, [10, 25, 50, 75, 90])
            final_cost = D_rhythm[-1, -1]
            
            logger.info(f"\nRhythm DTW Cost Distribution:")
            logger.info(f"10th percentile: {p10:.2f}")
            logger.info(f"25th percentile: {p25:.2f}")
            logger.info(f"Median: {p50:.2f}")
            logger.info(f"75th percentile: {p75:.2f}")
            logger.info(f"90th percentile: {p90:.2f}")
            logger.info(f"Final rhythm cost: {final_cost:.2f}")
            
            # Score rhythm using same scale as pitch
            if final_cost <= p10:
                rhythm_score = 1.0
            elif final_cost <= p25:
                rhythm_score = 0.8 + 0.2 * (p25 - final_cost) / (p25 - p10)
            elif final_cost <= p50:
                rhythm_score = 0.6 + 0.2 * (p50 - final_cost) / (p50 - p25)
            elif final_cost <= p75:
                rhythm_score = 0.4 + 0.2 * (p75 - final_cost) / (p75 - p50)
            elif final_cost <= p90:
                rhythm_score = 0.2 + 0.2 * (p90 - final_cost) / (p90 - p75)
            else:
                rhythm_score = max(0.0, 0.2 * (1 - (final_cost - p90) / p90))
            
            logger.info(f"Rhythm score: {rhythm_score:.2f}")
            return rhythm_score
            
        except Exception as e:
            logger.error(f"Error calculating rhythm score: {str(e)}")
            logger.error(f"Onset shapes - Original: {onset_obj[0].shape}, Cover: {onset_obj[1].shape}")
            return 0.5  # Return neutral score on error

    def compute_similarity(self, chroma_obj: List, onset_obj: List) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Compute DTW-based similarity between chromagrams and onsets."""
        try:
            # Downsample if sequences are too long
            chroma_orig_dtw, chroma_cover_dtw = self._prepare_dtw_sequences(chroma_obj)
            
            # Compute DTW for pitch
            D, wp = librosa.sequence.dtw(
                X=chroma_orig_dtw,
                Y=chroma_cover_dtw,
                metric='cosine'
            )
            
            # Compute rhythm score
            rhythm_score = self.compare_rhythm(onset_obj)
            
            return D, wp, rhythm_score
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise

    def _prepare_dtw_sequences(self, chroma_obj: List) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for DTW computation."""
        if chroma_obj[0].shape[1] > self.max_length or chroma_obj[1].shape[1] > self.max_length:
            factor = max(chroma_obj[0].shape[1], chroma_obj[1].shape[1]) // self.max_length + 1
            logger.info(f"Downsampling by factor of {factor} for DTW computation")
            return (chroma_obj[0][:, ::factor], chroma_obj[1][:, ::factor])
        return (chroma_obj[0], chroma_obj[1])

    def visualize_comparison(self, chroma_obj: List, D: np.ndarray, wp: np.ndarray) -> None:
        """Visualize the comparison results."""
        logger.info("Visualizing comparison results...")
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot original chroma
            plt.subplot(1, 3, 1)
            librosa.display.specshow(chroma_obj[0], y_axis='chroma', x_axis='time')
            plt.title('Original Chroma')
            plt.colorbar()
            
            # Plot cover chroma
            plt.subplot(1, 3, 2)
            librosa.display.specshow(chroma_obj[1], y_axis='chroma', x_axis='time')
            plt.title('Cover Chroma')
            plt.colorbar()
            
            # Plot DTW matrix
            plt.subplot(1, 3, 3)
            librosa.display.specshow(
                D / D.max(),
                x_axis='time',
                y_axis='time',
                cmap='gray_r'
            )
            plt.colorbar(label='Normalized distance')
            plt.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='r')
            plt.title('DTW Alignment Path')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing comparison: {str(e)}")
            raise

    def calculate_accuracy(self, D: np.ndarray, rhythm_score: float) -> float:
        """Calculate combined accuracy using both pitch and rhythm."""
        try:
            # Calculate pitch score using existing percentile method
            final_cost = D[-1, -1]
            p10, p25, p50, p75, p90 = np.percentile(D, [10, 25, 50, 75, 90])
            
            logger.info(f"\nPitch DTW Cost Distribution:")
            logger.info(f"10th percentile: {p10:.2f}")
            logger.info(f"25th percentile: {p25:.2f}")
            logger.info(f"Median: {p50:.2f}")
            logger.info(f"75th percentile: {p75:.2f}")
            logger.info(f"90th percentile: {p90:.2f}")
            logger.info(f"Final pitch cost: {final_cost:.2f}")
            
            # Calculate pitch score
            if final_cost <= p10:
                pitch_score = 1.0
            elif final_cost <= p25:
                pitch_score = 0.8 + 0.2 * (p25 - final_cost) / (p25 - p10)
            elif final_cost <= p50:
                pitch_score = 0.6 + 0.2 * (p50 - final_cost) / (p50 - p25)
            elif final_cost <= p75:
                pitch_score = 0.4 + 0.2 * (p75 - final_cost) / (p75 - p50)
            elif final_cost <= p90:
                pitch_score = 0.2 + 0.2 * (p90 - final_cost) / (p90 - p75)
            else:
                pitch_score = max(0.0, 0.2 * (1 - (final_cost - p90) / p90))
            
            logger.info(f"Pitch score: {pitch_score:.2f}")
            
            # Combine pitch and rhythm scores (weighted average)
            pitch_weight = 0.6  # Pitch is usually more important
            rhythm_weight = 0.4  # Rhythm contributes less to overall similarity
            
            combined_score = (pitch_weight * pitch_score) + (rhythm_weight * rhythm_score)
            
            logger.info(f"\nFinal Scores:")
            logger.info(f"Pitch Score: {pitch_score:.2f}")
            logger.info(f"Rhythm Score: {rhythm_score:.2f}")
            logger.info(f"Combined Score: {combined_score:.2f}")
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            raise
