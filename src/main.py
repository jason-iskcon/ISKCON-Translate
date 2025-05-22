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

# Process audio moved to TranscriptionEngine class

def main(video_file=None):
    """Minimal main function with hardcoded values."""
    # Use hardcoded video file if none provided
    if video_file is None:
        # Use a pre-downloaded video from cache
        # Assuming there's a video in the cache dir
        video_file = os.path.join(os.path.expanduser("~/.video_cache"), "3YhGU6vVBg8.mp4")
    
    try:
        logger.info(f"Starting synchronized video captioning with {video_file}")
        
        # Initialize components with improved configuration
        # Start at 5:25 (325 seconds) where English translation begins
        start_offset = 325.0  # Start 5:25 into the video
        with VideoSource(video_file, start_time=start_offset) as video_source, \
             TranscriptionEngine() as transcriber, \
             CaptionOverlay(
                 font_scale=1.5,  # Larger font for better readability
                 font_thickness=2,
                 font_color=(255, 255, 255),  # White text
                 bg_color=(0, 0, 0, 180),     # Semi-transparent black background
                 padding=15,                  # More padding around text
                 y_offset=50                  # Position from bottom of frame
             ) as caption_overlay:
            
            # Set the video start time for caption synchronization using actual playback start time
            video_start_time = video_source.playback_start_time
            caption_overlay.set_video_start_time(video_start_time)
            logger.info(f"[SYNC] Video start time set to: {video_start_time}")
            
            # Connect caption overlay to transcription engine
            transcriber.caption_overlay = caption_overlay
            
            # Start the transcription thread
            transcriber.start_transcription()
            
            # Test captions with offsets in seconds from start
            test_captions = [
                ("TEST 1: Overlay working", 1.0),
                ("TEST 2: Still alive!", 3.0),
                ("TEST 3: Quit with 'q'", 5.0),
            ]
            next_caption_index = 0
            last_caption_time = 0
            
            logger.info("All components initialized")
            
            # Get video information
            width, height, fps = video_source.get_video_info()
            target_frame_time = 1.0 / fps  # Time between frames
            logger.info(f"Video info: {width}x{height} @ {fps}fps")
            
            # Start with warm-up period (30 seconds)
            warm_up_duration = 30.0
            warm_up_end = time.time() + warm_up_duration
            logger.info(f"Starting {warm_up_duration:.1f} second warm-up period...")
            
            # Start audio processing thread (will start processing but not playing)
            audio_thread = threading.Thread(target=transcriber.process_audio, args=(video_source,))
            audio_thread.daemon = True
            audio_thread.start()
            
            # Wait for warm-up to complete
            while time.time() < warm_up_end:
                remaining = warm_up_end - time.time()
                if remaining % 5 < 0.1:  # Log every ~5 seconds
                    logger.info(f"Warm-up: {remaining:.1f}s remaining...")
                time.sleep(0.1)
            
            logger.info("Warm-up complete, starting playback")
            
            # Create window for display (only after warm-up)
            window_name = "Video with Captions"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width, height)
            
            # Schedule test captions with relative timestamps
            # Using video-relative timestamps (0 = start of video)
            logger.info(f"[DEBUG] Video start time: {video_start_time}")
            
            # Signal video source that warm-up is complete
            video_source.warm_up_complete.set()

            # Set playback start time for correct timestamping in transcription
            transcriber.playback_start_time = video_source.playback_start_time
            logger.info(f"[SYNC] Set transcriber.playback_start_time = {transcriber.playback_start_time:.2f}")
            
            # Wait a brief moment after warm-up to ensure timing is stable
            time.sleep(0.5)
            
            # Get the current time after warm-up is complete
            current_playback_time = time.time() - video_start_time
            logger.info("\n=== SCHEDULING TEST CAPTIONS ===")
            logger.info(f"System time: {time.time()}")
            logger.info(f"Video start time: {video_start_time}")
            logger.info(f"Current playback time: {current_playback_time:.2f}s")
            
            # Schedule test captions relative to current playback time
            test_offsets = [1.0, 3.0, 5.0]  # 1s, 3s, and 5s from now
            
            for (caption_text, _), offset in zip(test_captions, test_offsets):
                caption_time = current_playback_time + offset
                logger.info(f"\nAdding caption: '{caption_text}' at {offset:.1f}s from now")
                logger.info(f"  - Will appear at system time: {time.time() + offset:.2f}")
                logger.info(f"  - Relative to video start: {caption_time:.2f}s")
                
                caption_overlay.add_caption(
                    text=caption_text,
                    timestamp=caption_time,  # Absolute time when caption should appear
                    duration=10.0,  # 10 seconds duration
                    is_absolute=False  # Using relative timestamp from video start
                )
                logger.info(f"[TEST] Scheduled caption: '{caption_text}' for {caption_time:.2f}s from video start")

            # Start audio playback thread now that warm-up is complete
            if video_source.audio_file:
                logger.info("Starting audio playback thread")
                video_source.audio_playing = True
                video_source.audio_thread = threading.Thread(
                    target=video_source._play_audio,
                    args=(video_source.audio_file,)
                )
                video_source.audio_thread.daemon = True
                video_source.audio_thread.start()
            
            # Small delay to ensure video thread is ready
            time.sleep(0.5)
            
            # Simple playback control variables
            paused = False
            frame_count = 0
            frame_buffer = []
            max_buffer_size = 30  # Increased buffer size for smoother playback
            
            # Simple timing variables
            next_frame_time = time.time()
            current_video_time = 0.0
            
            # Pre-buffer frames before starting playback
            logger.info("Pre-buffering frames")
            while len(frame_buffer) < max_buffer_size:
                frame_data = video_source.get_frame()
                if frame_data is not None:
                    frame_buffer.append(frame_data)
                else:
                    time.sleep(0.01)
                    
            logger.info(f"Starting playback with {len(frame_buffer)} frames in buffer")
            
            # Main playback loop with synchronization
            last_frame_time = time.time()
            next_frame_time = last_frame_time
            
            while True:
                current_time = time.time()
                
                # Get more frames if buffer is low (in a non-blocking way)
                if len(frame_buffer) < max_buffer_size // 2:
                    try:
                        frame_data = video_source.get_frame()
                        if frame_data is not None:
                            frame_buffer.append(frame_data)
                    except:
                        pass  # Silently handle any frame fetch errors
                
                # Process frame when it's time
                if not paused and frame_buffer and current_time >= next_frame_time:
                    # Get frame with its timestamp
                    frame, frame_timestamp = frame_buffer.pop(0)
                    
                    # Get current audio position for synchronization
                    with video_source.audio_position_lock:
                        current_audio_time = video_source.audio_position
                    
                    # Use audio time for synchronization if available
                    current_video_time = current_audio_time if video_source.audio_playing else frame_timestamp
                    frame_count += 1
                    
                    # Calculate time delta and next frame time
                    now = time.time()
                    frame_delay = now - last_frame_time
                    last_frame_time = now
                    next_frame_time = now + max(0, target_frame_time - frame_delay)
                    
                    try:
                        # Get any available transcription
                        while True:  # Process all available transcriptions
                            try:
                                transcription = transcriber.get_transcription()
                                if not transcription:
                                    logger.debug("No more transcriptions in queue")
                                    break
                                
                                try:
                                    text = transcription['text'].strip()
                                    start_time = transcription['timestamp']
                                    end_time = transcription.get('end_time', start_time + 3.0)  # Default 3s duration
                                    duration = end_time - start_time
                                    
                                    # Log the transcription with current video time for debugging
                                    logger.info(f"[TRANSCRIPTION] Received: '{text}' at {start_time:.2f}s (video: {video_source.get_current_time():.2f}s)")
                                    
                                    # Ensure start_time is relative to video start
                                    if start_time > video_start_time:  # Likely an absolute timestamp
                                        start_time = start_time - video_start_time
                                        logger.info(f"[TIMING] Converted absolute timestamp to relative: {start_time:.2f}s")
                                    
                                    # Add the caption with relative time
                                    caption_text = text
                                    if caption_text:  # Only add non-empty captions
                                        logger.info(f"[CAPTION] Adding: {caption_text!r} at {start_time:.2f}s for {duration:.1f}s")
                                        try:
                                            # Ensure we're using relative timestamps (0-based from video start)
                                            relative_start = start_time
                                            logger.info(f"  - Using relative start time: {relative_start:.2f}s")
                                            
                                            caption_overlay.add_caption(
                                                caption_text,
                                                timestamp=relative_start,
                                                duration=duration,
                                                is_absolute=False  # Explicitly using relative timestamps
                                            )
                                            logger.info("[CAPTION] Added successfully")
                                        except Exception as e:
                                            logger.error(f"Error adding caption: {e}", exc_info=True)
                                    
                                except Exception as e:
                                    logger.error(f"Error processing transcription: {e}", exc_info=True)
                                    break
                                    
                            except queue.Empty:
                                break
                        
                        # Calculate current relative time since video start
                        current_time = time.time()
                        current_relative_time = current_time - video_start_time
                        
                        # Log timing info every 30 frames (~1 second at 30fps)
                        if frame_count % 30 == 0:
                            logger.info("\n=== FRAME TIMING ===")
                            logger.info(f"System time: {current_time}")
                            logger.info(f"Video start time: {video_start_time}")
                            logger.info(f"Current relative time: {current_relative_time:.2f}s")
                            logger.info(f"Frame: {frame_count}")
                            
                            # Log all captions and their timing
                            if hasattr(caption_overlay, 'captions') and caption_overlay.captions:
                                logger.info("\n=== CAPTIONS ===")
                                for i, cap in enumerate(caption_overlay.captions):
                                    active = "ACTIVE" if cap['start_time'] <= current_relative_time <= cap['end_time'] else "     "
                                    logger.info(f"{active} [{i}] '{cap['text'][:30]}...' | Start: {cap['start_time']:.2f}s | End: {cap['end_time']:.2f}s | Now: {current_relative_time:.2f}s")
                            else:
                                logger.info("\n=== NO CAPTIONS LOADED ===")
                        
                        # Debug: Log active captions
                        if frame_count % 30 == 0:  # Log every second at 30fps
                            active_captions = [
                                c for c in caption_overlay.captions 
                                if c['start_time'] <= current_relative_time <= c['end_time']
                            ]
                            
                            if active_captions:
                                for cap in active_captions:
                                    logger.info(f"[CAPTION] Active: '{cap['text']}' ({cap['start_time']:.2f}-{cap['end_time']:.2f}) at time {current_relative_time:.2f}")
                            else:
                                logger.info(f"[DEBUG] No active captions at time {current_relative_time:.2f}")
                                if caption_overlay.captions:
                                    logger.info("[DEBUG] Available captions: " + 
                                        ", ".join([f"'{c['text'][:20]}...' ({c['start_time']:.2f}-{c['end_time']:.2f})" 
                                        for c in caption_overlay.captions[:5]]))
                        
                        # Apply captions to frame using relative time and frame count
                        frame_copy = frame.copy()
                        frame_with_captions = caption_overlay.overlay_captions(
                            frame=frame_copy,
                            current_time=current_relative_time,  # Use relative time
                            frame_count=frame_count  # Pass frame count for logging
                        )
                        
                        # Debug: Log the current time and caption state
                        if frame_count % 30 == 0:  # Log every second at 30fps
                            logger.info(f"[DEBUG] Frame {frame_count}: relative_time={current_relative_time:.2f}s, "
                                      f"video_time={video_source.get_current_time():.2f}s")
                        
                        # Show frame number and relative time for debugging
                        debug_text = f"Frame: {frame_count}, RelTime: {current_relative_time:.2f}s"
                        cv2.putText(
                            frame_with_captions,
                            debug_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),  # Red text for better visibility
                            2,  # Thicker text
                            cv2.LINE_AA
                        )
                        
                        # Add active caption count for debugging
                        active_caption_count = len([
                            c for c in caption_overlay.captions 
                            if c['start_time'] <= current_relative_time <= c['end_time']
                        ])
                        debug_time_text = f"Captions: {active_caption_count}"
                        cv2.putText(
                            frame_with_captions,
                            debug_time_text,
                            (10, 60),  # Positioned below the frame counter
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),  # Yellow text
                            2,
                            cv2.LINE_AA
                        )
                        logger.debug(f"Added debug text: {debug_text}")
                        
                        # Display ONLY the frame with captions (never the raw frame)
                        cv2.imshow(window_name, frame_with_captions)
                        
                        # Store the last frame with captions for display
                        last_frame = frame_with_captions
                    except Exception as e:
                        logger.warning(f"Error processing frame: {e}")
                
                # Check for key presses (blocking with 1ms timeout)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # If a key was pressed
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        logger.info("Quit requested by user")
                        break
                    elif key == ord('p') or key == 32:  # 'p' or SPACE
                        paused = not paused
                        status = "paused" if paused else "resumed"
                        logger.info(f"Playback {status}")
                        if not paused:
                            next_frame_time = time.time()
                
                # Adaptive sleep to reduce CPU usage
                sleep_time = max(0, next_frame_time - time.time() - 0.001)
                if sleep_time > 0.001:
                    time.sleep(min(0.01, sleep_time))
            
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
