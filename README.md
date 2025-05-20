# Video Captioning System with Synchronization - MVP

This MVP implements a simplified video captioning system with a focus on **synchronized** audio, video, and caption overlay. The implementation follows the principles outlined in the synchronization research documents.

## Key Synchronization Features

1. **Timestamp-Based Synchronization**: All components (video, audio, captions) are synchronized using a common timeline based on presentation timestamps (PTS).

2. **Buffer Management**: Intelligent buffer management to ensure smooth video playback while maintaining audio-video sync.

3. **Coordinated Processing Pipeline**: Separate threads for audio and video processing, carefully coordinated through timestamps.

4. **Caption Timing**: Captions are displayed based on their original audio timestamps, ensuring proper alignment with video content.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (install using `pip install -r requirements.txt`)
- Pre-downloaded Whisper model in the Hugging Face cache (~/.cache/huggingface/hub)

### Running the MVP

1. To caption a YouTube video:
   ```
   python run.py --source "https://www.youtube.com/watch?v=VIDEO_ID"
   ```

2. To caption a local video file:
   ```
   python run.py --source "/path/to/video.mp4"
   ```

The application will display the video with synchronized captions in a window. Press 'p' to pause/resume and 'q' to quit.

## Implementation Details

### Video Source (`src/video_source.py`)
- Extracts video frames with precise timestamps
- Provides audio chunks with matching timestamps
- Uses simplified YouTube download and caching

### Transcription Engine (`src/transcription.py`) 
- Processes audio in chunks with timestamp tracking
- Maintains buffer continuity for better transcription
- Returns transcriptions with original timestamps for synchronization

### Caption Overlay (`src/caption_overlay.py`)
- Maintains captions with start/end timestamps
- Displays captions synchronized with video PTS
- Removes expired captions based on timing

### Main Application (`src/main.py`)
- Orchestrates the synchronized playback of all components
- Maintains buffer management for smooth playback
- Tracks and displays timing information for debugging

## Simplifications for MVP

1. Uses OpenCV instead of PyAV (simpler but less accurate A/V sync)
2. Extracts audio in a simplified manner (real implementation would use proper audio demuxing)
3. Limited to English language only (no Sanskrit support in MVP)
4. Uses pre-downloaded Whisper models from cache only
5. Basic error handling and logging
