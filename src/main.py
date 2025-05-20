import cv2
import sys
import os
import time
import threading
import logging

from .video_source import VideoSource
from .transcription import TranscriptionEngine
from .caption_overlay import CaptionOverlay

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_audio(video_source, transcriber):
    """Process audio from video source and send to transcriber."""
    logger.info("Starting audio processing thread")
    last_time = 0
    
    try:
        while video_source.is_running:
            # Check current audio position
            with video_source.audio_position_lock:
                current = video_source.audio_position
            # Process only if we've advanced by 2+ seconds
            if current >= last_time + 2.0:
                audio_data = video_source.get_audio_chunk()
                if audio_data is not None:
                    transcriber.add_audio_segment(audio_data)
                    last_time = current
            # Small delay
            time.sleep(0.1)
    except Exception as e:
        logger.error(f"Error in audio processing: {e}")
        
    logger.info("Audio processing stopped")

def main(video_file=None):
    """Minimal main function with hardcoded values."""
    # Use hardcoded video file if none provided
    if video_file is None:
        # Use a pre-downloaded video from cache
        # Assuming there's a video in the cache dir
        video_file = os.path.join(os.path.expanduser("~/.video_cache"), "3YhGU6vVBg8.mp4")
    
    try:
        logger.info(f"Starting synchronized video captioning with {video_file}")
        
        # Initialize components with minimal configuration
        with VideoSource(video_file) as video_source, \
             TranscriptionEngine() as transcriber, \
             CaptionOverlay() as caption_overlay:
            
            logger.info("All components initialized")
            
            # Get video information
            width, height, fps = video_source.get_video_info()
            target_frame_time = 1.0 / fps  # Time between frames
            logger.info(f"Video info: {width}x{height} @ {fps}fps")
            
            # Create window for display
            window_name = "Video with Captions"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width, height)
            
            # Start audio processing thread
            audio_thread = threading.Thread(target=process_audio, args=(video_source, transcriber))
            audio_thread.daemon = True
            audio_thread.start()
            
            # Simple playback control variables
            paused = False
            frame_count = 0
            frame_buffer = []
            max_buffer_size = 20  # Hardcoded small buffer size for simplicity
            
            # Simple timing variables
            next_frame_time = time.time()
            current_video_time = 0.0
            
            # Pre-buffer some frames before starting
            logger.info("Pre-buffering frames")
            prebuffer_count = max_buffer_size // 2
            while len(frame_buffer) < prebuffer_count:
                frame_data = video_source.get_frame()
                if frame_data is not None:
                    frame_buffer.append(frame_data)
                else:
                    time.sleep(0.01)
                    
            logger.info(f"Starting playback with {len(frame_buffer)} frames in buffer")
            
            # Main playback loop with synchronization
            while True:
                current_time = time.time()
                
                # Get more frames if buffer is low
                if not paused and len(frame_buffer) < max_buffer_size // 2:
                    for _ in range(min(5, max_buffer_size - len(frame_buffer))):
                        frame_data = video_source.get_frame()
                        if frame_data is None:
                            break
                        frame_buffer.append(frame_data)
                
                # Process frame when it's time
                if current_time >= next_frame_time and not paused:
                    # Get frame with its timestamp
                    if frame_buffer:
                        frame, frame_timestamp = frame_buffer.pop(0)
                        current_video_time = frame_timestamp  # Update current PTS
                        frame_count += 1
                    else:
                        time.sleep(0.01)
                        continue
                    
                    # Calculate next frame time
                    next_frame_time = max(current_time, next_frame_time + target_frame_time)
                    
                    # Get any available transcription
                    transcription = transcriber.get_transcription()
                    if transcription:
                        # Add caption with timestamp for synchronization
                        caption_overlay.add_caption(
                            text=transcription["text"].strip(),
                            timestamp=transcription["timestamp"]
                        )
                    
                    # Overlay captions synchronized with current video time
                    display_frame = caption_overlay.overlay_captions(frame, current_video_time)
                    
                    # Show frame number and time for debugging
                    cv2.putText(
                        display_frame,
                        f"Frame: {frame_count}, Time: {current_video_time:.2f}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1
                    )
                    
                    # Display the frame
                    cv2.imshow(window_name, display_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    if not paused:
                        next_frame_time = time.time()
                
                # Efficient sleep if we have time to spare
                if next_frame_time > current_time + 0.01:
                    time.sleep(0.01)
                else:
                    time.sleep(0.001)
            
            logger.info(f"Playback ended after {frame_count} frames")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    
    # Import here to avoid circular imports
    from sys import argv
    
    # Use command line arg if provided, otherwise use default
    video_file = argv[1] if len(argv) > 1 else None
    main(video_file)
