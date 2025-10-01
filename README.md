# audio-speech-segmentation
"Audio speech segmentation system that extracts and segments speech from video/audio files"
# Audio Speech Segmentation System

Extracts audio from video files, detects speech segments, and exports individual clips.

## Features
- ✅ Extracts audio from video/audio files (MP4, AVI, MOV, MP3, WAV, etc.)
- ✅ Detects speech segments using energy-based analysis
- ✅ Exports segments as separate audio files
- ✅ Generates JSON timestamps and summary reports
- ✅ Progress tracking with detailed console output

## Requirements
```bash
pip install pydub numpy scipy moviepy tqdm

python audio_segmenter.py input_video.mp4

## Output
- `extracted_audio.wav` - Full extracted audio
- `speech_timestamps.json` - Speech segment timestamps
- `segments/` - Individual speech clips

## Example Results
- Total duration: 12.59 seconds
- Segments detected: 2
- Segment 1: 8.28 seconds
- Segment 2: 1.0 seconds
