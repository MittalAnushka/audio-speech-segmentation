"""
Audio Speech Segmentation System
Extracts audio from video, detects speech segments, and exports clips.

Requirements:
pip install pydub numpy scipy moviepy tqdm
"""

import os
import json
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.io import wavfile
import argparse
from tqdm import tqdm
import sys
import traceback


class AudioSegmenter:
    def __init__(self, output_dir="output"):
        """Initialize the audio segmenter with output directory."""
        self.output_dir = output_dir
        self.create_directories()
        
    def create_directories(self):
        """Create necessary output directories."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "segments"), exist_ok=True)
            print(f"✓ Output directories created: {os.path.abspath(self.output_dir)}")
        except Exception as e:
            print(f"✗ Error creating directories: {e}")
            sys.exit(1)
        
    def extract_audio(self, input_file):
        """
        Extract audio from video or load audio file.
        
        Args:
            input_file: Path to input video or audio file
            
        Returns:
            Path to extracted audio file
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"\n{'='*60}")
        print(f"STEP 1: Extracting Audio")
        print(f"{'='*60}")
        print(f"Input file: {input_file}")
        
        file_ext = os.path.splitext(input_file)[1].lower()
        audio_output = os.path.join(self.output_dir, "extracted_audio.wav")
        
        # Define supported formats
        video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        audio_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']
        
        try:
            if file_ext in video_formats:
                print("✓ Detected video file. Extracting audio track...")
                from moviepy.editor import VideoFileClip
                
                video = VideoFileClip(input_file)
                
                if video.audio is None:
                    raise ValueError("Video file has no audio track!")
                
                # Extract audio with progress
                video.audio.write_audiofile(
                    audio_output,
                    fps=16000,
                    nbytes=2,
                    codec='pcm_s16le',
                    verbose=False,
                    logger=None
                )
                video.close()
                
            elif file_ext in audio_formats:
                print("✓ Detected audio file. Converting to standard format...")
                audio = AudioSegment.from_file(input_file)
                
                # Convert to mono, 16kHz
                print(f"  Original: {audio.channels} channel(s), {audio.frame_rate}Hz")
                audio = audio.set_channels(1).set_frame_rate(16000)
                print(f"  Converted: 1 channel (mono), 16000Hz")
                
                audio.export(audio_output, format="wav")
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}\n"
                               f"Supported video: {', '.join(video_formats)}\n"
                               f"Supported audio: {', '.join(audio_formats)}")
            
            # Get file info
            file_size = os.path.getsize(audio_output) / (1024 * 1024)
            audio_check = AudioSegment.from_wav(audio_output)
            duration = len(audio_check) / 1000.0
            
            print(f"✓ Audio extracted successfully!")
            print(f"  Output: {audio_output}")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Size: {file_size:.2f} MB")
            
            return audio_output
            
        except Exception as e:
            print(f"✗ Error extracting audio: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def detect_speech_segments(self, audio_file, min_silence_len=500, 
                               silence_thresh=-40, min_speech_len=300):
        """
        Detect speech segments in audio using energy-based approach.
        
        Args:
            audio_file: Path to audio file
            min_silence_len: Minimum silence length in ms to consider as gap
            silence_thresh: Silence threshold in dBFS
            min_speech_len: Minimum speech segment length in ms
            
        Returns:
            List of speech segments with start/end timestamps in seconds
        """
        print(f"\n{'='*60}")
        print(f"STEP 2: Detecting Speech Segments")
        print(f"{'='*60}")
        
        try:
            # Load audio
            print("Loading audio file...")
            audio = AudioSegment.from_wav(audio_file)
            duration = len(audio) / 1000.0
            print(f"✓ Audio loaded: {duration:.2f} seconds")
            
            # Display detection parameters
            print(f"\nDetection parameters:")
            print(f"  - Minimum silence length: {min_silence_len}ms")
            print(f"  - Silence threshold: {silence_thresh}dBFS")
            print(f"  - Minimum speech length: {min_speech_len}ms")
            
            # Detect non-silent chunks with progress
            print("\nAnalyzing audio for speech...")
            nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                seek_step=10
            )
            
            print(f"✓ Found {len(nonsilent_ranges)} potential speech segments")
            
            # Convert to seconds and filter short segments
            speech_segments = []
            filtered_count = 0
            
            for start_ms, end_ms in nonsilent_ranges:
                duration_ms = end_ms - start_ms
                if duration_ms >= min_speech_len:
                    segment = {
                        "start": round(start_ms / 1000.0, 2),
                        "end": round(end_ms / 1000.0, 2),
                        "duration": round(duration_ms / 1000.0, 2)
                    }
                    speech_segments.append(segment)
                else:
                    filtered_count += 1
            
            print(f"✓ Filtered out {filtered_count} segments shorter than {min_speech_len}ms")
            print(f"✓ Final speech segments: {len(speech_segments)}")
            
            if speech_segments:
                total_speech_time = sum(seg["duration"] for seg in speech_segments)
                print(f"\nStatistics:")
                print(f"  - Total speech time: {total_speech_time:.2f}s")
                print(f"  - Speech coverage: {(total_speech_time/duration)*100:.1f}%")
                print(f"  - Average segment length: {total_speech_time/len(speech_segments):.2f}s")
            
            # Save timestamps to JSON
            timestamps_file = os.path.join(self.output_dir, "speech_timestamps.json")
            
            output_data = {
                "total_duration": round(duration, 2),
                "segment_count": len(speech_segments),
                "parameters": {
                    "min_silence_len": min_silence_len,
                    "silence_thresh": silence_thresh,
                    "min_speech_len": min_speech_len
                },
                "segments": speech_segments
            }
            
            with open(timestamps_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n✓ Timestamps saved to: {timestamps_file}")
            
            return speech_segments
            
        except Exception as e:
            print(f"✗ Error detecting speech segments: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def export_segments(self, audio_file, segments, output_format="wav"):
        """
        Export individual speech segments as separate audio files.
        
        Args:
            audio_file: Path to source audio file
            segments: List of segment dictionaries with start/end times
            output_format: Output format (wav, mp3, etc.)
        """
        print(f"\n{'='*60}")
        print(f"STEP 3: Exporting Audio Segments")
        print(f"{'='*60}")
        
        if not segments:
            print("✗ No segments to export!")
            return
        
        try:
            print("Loading source audio...")
            audio = AudioSegment.from_wav(audio_file)
            segments_dir = os.path.join(self.output_dir, "segments")
            
            print(f"✓ Exporting {len(segments)} segments to {output_format.upper()} format...")
            
            # Export with progress bar
            for idx, segment in enumerate(tqdm(segments, desc="Exporting", unit="segment"), 1):
                start_ms = int(segment["start"] * 1000)
                end_ms = int(segment["end"] * 1000)
                
                # Extract segment
                audio_segment = audio[start_ms:end_ms]
                
                # Export with zero-padded naming
                filename = f"segment_{idx:03d}.{output_format}"
                output_path = os.path.join(segments_dir, filename)
                
                # Export based on format
                if output_format == "mp3":
                    audio_segment.export(output_path, format="mp3", bitrate="192k")
                else:
                    audio_segment.export(output_path, format=output_format)
            
            print(f"\n✓ All segments exported successfully!")
            print(f"  Location: {os.path.abspath(segments_dir)}")
            
            # Create segment summary file
            summary_file = os.path.join(segments_dir, "segments_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("Audio Segments Summary\n")
                f.write("=" * 60 + "\n\n")
                for idx, segment in enumerate(segments, 1):
                    f.write(f"Segment {idx:03d}:\n")
                    f.write(f"  Start: {segment['start']:.2f}s\n")
                    f.write(f"  End: {segment['end']:.2f}s\n")
                    f.write(f"  Duration: {segment['duration']:.2f}s\n")
                    f.write(f"  File: segment_{idx:03d}.{output_format}\n\n")
            
            print(f"✓ Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"✗ Error exporting segments: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def process(self, input_file, min_silence_len=500, silence_thresh=-40, 
                min_speech_len=300, output_format="wav"):
        """
        Complete processing pipeline: extract, detect, segment.
        
        Args:
            input_file: Path to input video or audio file
            min_silence_len: Minimum silence length in ms
            silence_thresh: Silence threshold in dBFS
            min_speech_len: Minimum speech segment length in ms
            output_format: Output format for segments
        """
        print("\n" + "=" * 60)
        print("🎵 AUDIO SPEECH SEGMENTATION PIPELINE 🎵")
        print("=" * 60)
        
        try:
            # Step 1: Extract audio
            audio_file = self.extract_audio(input_file)
            
            # Step 2: Detect speech segments
            segments = self.detect_speech_segments(
                audio_file,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                min_speech_len=min_speech_len
            )
            
            # Step 3: Export segments
            if segments:
                self.export_segments(audio_file, segments, output_format)
            else:
                print("\n⚠ Warning: No speech segments detected!")
                print("  Try adjusting the parameters:")
                print("  - Lower --silence-thresh (e.g., -50)")
                print("  - Increase --min-silence (e.g., 700)")
                print("  - Decrease --min-speech (e.g., 200)")
            
            # Final summary
            print("\n" + "=" * 60)
            print("✓ PROCESSING COMPLETE!")
            print("=" * 60)
            print(f"\n📁 Output directory: {os.path.abspath(self.output_dir)}")
            print(f"   ├── extracted_audio.wav")
            print(f"   ├── speech_timestamps.json")
            print(f"   └── segments/")
            print(f"       ├── segment_001.{output_format}")
            print(f"       ├── segment_002.{output_format}")
            print(f"       ├── ... ({len(segments)} files total)")
            print(f"       └── segments_summary.txt")
            print("\n" + "=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n✗ Process interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Fatal error: {e}")
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio, detect speech, and create segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4
  %(prog)s audio.mp3 -o my_output
  %(prog)s video.mp4 --min-silence 700 --silence-thresh -35
  %(prog)s audio.wav --format mp3
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input video or audio file"
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--min-silence",
        type=int,
        default=500,
        help="Minimum silence length in ms (default: 500)"
    )
    parser.add_argument(
        "--silence-thresh",
        type=int,
        default=-40,
        help="Silence threshold in dBFS, lower = more sensitive (default: -40)"
    )
    parser.add_argument(
        "--min-speech",
        type=int,
        default=300,
        help="Minimum speech segment length in ms (default: 300)"
    )
    parser.add_argument(
        "--format",
        choices=["wav", "mp3", "flac", "ogg"],
        default="wav",
        help="Output format for segments (default: wav)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Audio Speech Segmentation System v1.0"
    )
    
    args = parser.parse_args()
    
    # Create segmenter and process
    segmenter = AudioSegmenter(output_dir=args.output)
    segmenter.process(
        args.input_file,
        min_silence_len=args.min_silence,
        silence_thresh=args.silence_thresh,
        min_speech_len=args.min_speech,
        output_format=args.format
    )


if __name__ == "__main__":
    main()