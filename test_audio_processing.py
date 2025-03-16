import numpy as np
import librosa
from unittest import mock
import unittest
import time
import cProfile
import pstats
import psutil
import os
from helper_functions import *

class TestAudioProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize shared test data once for all tests"""
        print("Initializing test data...")
        
        # Basic test parameters
        cls.sample_length = 100
        cls.n_chroma = 12
        cls.sample_rate = 22050
        cls.hop_length = 1024
        
        # Create sample data
        cls.sample_chroma1 = np.random.rand(cls.n_chroma, cls.sample_length)
        cls.sample_chroma2 = np.random.rand(cls.n_chroma, cls.sample_length)
        
        # Create sample onset data
        cls.sample_onset1 = np.random.rand(cls.sample_length)
        cls.sample_onset2 = np.random.rand(cls.sample_length)
        
        print("Test data initialized!")

    def setUp(self):
        """Runs before each test"""
        print(f"\nStarting test: {self._testMethodName}")
        self.processor = AudioProcessor()

    def tearDown(self):
        """Runs after each test"""
        print(f"Completed test: {self._testMethodName}\n")

    def test_dtw_performance(self):
        """Test the performance of DTW computation"""
        print("Testing DTW performance with small sample...")
        start_time = time.time()
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run DTW
        D, wp = librosa.sequence.dtw(
            X=self.sample_chroma1,
            Y=self.sample_chroma2,
            metric='cosine'
        )
        
        profiler.disable()
        end_time = time.time()
        
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(10)
        
        print(f"DTW computation took {end_time - start_time:.2f} seconds")
        print(f"DTW matrix shape: {D.shape}")
        
        self.assertLess(end_time - start_time, 5.0, "DTW computation took too long")

    def test_rhythm_comparison(self):
        """Test the rhythm comparison functionality"""
        print("Testing rhythm comparison...")
        start_time = time.time()
        
        onset_obj = [self.sample_onset1, self.sample_onset2]
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Test rhythm comparison
        rhythm_score = self.processor.compare_rhythm(onset_obj)
        
        profiler.disable()
        end_time = time.time()
        
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(10)
        
        print(f"Rhythm comparison took {end_time - start_time:.2f} seconds")
        print(f"Rhythm score: {rhythm_score:.2f}")
        
        self.assertGreaterEqual(rhythm_score, 0.0)
        self.assertLessEqual(rhythm_score, 1.0)
        self.assertLess(end_time - start_time, 5.0, "Rhythm comparison took too long")

    def test_combined_scoring(self):
        """Test the combined pitch and rhythm scoring"""
        print("Testing combined scoring...")
        
        # Create sample DTW matrix
        D = np.random.rand(50, 50)
        rhythm_score = 0.75  # Sample rhythm score
        
        # Test combined scoring
        combined_score = self.processor.calculate_accuracy(D, rhythm_score)
        
        print(f"Combined score: {combined_score:.2f}")
        
        self.assertGreaterEqual(combined_score, 0.0)
        self.assertLessEqual(combined_score, 1.0)

    def test_feature_extraction(self):
        """Test the feature extraction performance"""
        print("Testing feature extraction...")
        
        # Create sample audio data
        duration = 2  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        sample_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        start_time = time.time()
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Extract features
        chroma = librosa.feature.chroma_cqt(
            y=sample_audio,
            sr=self.sample_rate,
            norm=2,
            hop_length=self.hop_length
        )
        
        profiler.disable()
        end_time = time.time()
        
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(10)
        
        print(f"Feature extraction took {end_time - start_time:.2f} seconds")
        print(f"Chroma shape: {chroma.shape}")
        
        self.assertLess(end_time - start_time, 5.0, "Feature extraction took too long")

    def test_memory_usage(self):
        """Test memory usage during processing"""
        print("Testing memory usage...")
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Run complete analysis
        chroma_obj = [self.sample_chroma1, self.sample_chroma2]
        onset_obj = [self.sample_onset1, self.sample_onset2]
        
        D, wp, rhythm_score = self.processor.compute_similarity(chroma_obj, onset_obj)
        accuracy = self.processor.calculate_accuracy(D, rhythm_score)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        print(f"Memory usage: {memory_used:.2f} MB")
        self.assertLess(memory_used, 1000, "Memory usage exceeded 1GB")

if __name__ == '__main__':
    print("Starting audio processing tests...")
    unittest.main(verbosity=2)

