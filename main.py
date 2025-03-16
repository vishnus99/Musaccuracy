from helper_functions import AudioProcessor
import logging
import os

def main():
    # Set up logging to see the percentile information
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = AudioProcessor()
    
    # Process files with explicit paths
    audio_dir = "Audio Files"
    my_audio_dir = "My Audio Files"
    
    mp3_file = "theboxeroriginal.mp3"
    cover_file = "theboxercover.mp3"
    
    # Verify files exist
    if not os.path.exists(os.path.join(audio_dir, mp3_file)):
        raise FileNotFoundError(f"Original audio file not found in {audio_dir}")
    if not os.path.exists(os.path.join(my_audio_dir, cover_file)):
        raise FileNotFoundError(f"Cover audio file not found in {my_audio_dir}")
    
    # Split audio if separated files don't exist
    original_separated_path = os.path.join("separated", "htdemucs", mp3_file.rsplit('.', 1)[0], "vocals.wav")
    cover_separated_path = os.path.join("separated", "htdemucs", cover_file.rsplit('.', 1)[0], "vocals.wav")
    if not os.path.exists(original_separated_path) or not os.path.exists(cover_separated_path):
        print(f"Splitting original audio file: {mp3_file}")
        processor.split_audio_file(mp3_file, cover_file)
        print("Split complete!")
    else:
        print("Using existing separated audio files...")
    
    # Load and process
    print("Loading audio files...")
    original_audio, cover_audio = processor.load_audio_files(mp3_file, "vocals", cover_file)
    
    print("Extracting features...")
    chroma, onset = processor.extract_features(original_audio, cover_audio)
    
    # Compute similarity and visualize
    print("\nComputing similarity...")
    D, wp, rhythm_score = processor.compute_similarity(chroma, onset)
    processor.visualize_comparison(chroma, D, wp)
    accuracy = processor.calculate_accuracy(D, rhythm_score)
    
    print("\nAnalysis Complete")
    print("=" * 50)
    print(f"Final accuracy score: {accuracy:.2f}")
    print("Score interpretation:")
    print("  1.0 - 0.8: Excellent match")
    print("  0.8 - 0.6: Good match")
    print("  0.6 - 0.4: Fair match")
    print("  0.4 - 0.2: Poor match")
    print("  0.2 - 0.0: Very poor match")
    print("=" * 50)

if __name__ == "__main__":
    main()