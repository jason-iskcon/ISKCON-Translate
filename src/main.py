import cv2
import sys
import os
import time
import threading
import logging

# Import with try-except to handle both direct execution and module import
try:
    from video_source import VideoSource
    from transcription import TranscriptionEngine
    from caption_overlay import CaptionOverlay
    from logging_utils import get_logger
    from core.argument_parser import parse_arguments
    from core.app_initializer import initialize_app
    from core.video_runner import VideoRunner
    from clock import CLOCK
except ImportError:
    from .video_source import VideoSource
    from .transcription import TranscriptionEngine
    from .caption_overlay import CaptionOverlay
    from .logging_utils import get_logger
    from .core.argument_parser import parse_arguments
    from .core.app_initializer import initialize_app
    from .core.video_runner import VideoRunner
    from .clock import CLOCK

# Get logger instance
logger = get_logger(__name__)

# Process audio moved to TranscriptionEngine class

def main(video_file=None):
    """Main function for synchronized video captioning."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize application with logging
    logs_dir, log_file = initialize_app(log_level=args.log_level)
    
    # Use provided source file (from either positional or --source) or fallback
    video_file = args.source or args.source_arg or video_file
    if video_file is None:
        # Fallback to default video in cache
        video_file = os.path.join(os.path.expanduser("~/.video_cache"), "3YhGU6vVBg8.mp4")
    
    try:
        logger.info(f"Starting synchronized video captioning with {video_file}")
        
        # Initialize components with improved configuration
        # Start at 5:25 (325 seconds) where English translation begins
        start_offset = 325.0  # Start 5:25 into the video
        with VideoSource(video_file, start_time=start_offset) as video_source, \
             TranscriptionEngine(warm_up_complete_event=video_source.warm_up_complete) as transcriber, \
             CaptionOverlay(
                 font_scale=1.5,  # Larger font for better readability
                 font_thickness=2,
                 font_color=(255, 255, 255),  # White text
                 bg_color=(0, 0, 0, 180),     # Semi-transparent black background
                 padding=15,                  # More padding around text
                 y_offset=50                  # Position from bottom of frame
             ) as caption_overlay:
            
            # Caption overlay video start time will be set after singleton clock initialization
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
            
            logger.info("All components initialized")
            
            # BUILD VERIFICATION BANNER - Confirm queue sizes and runtime version
            git_commit = os.environ.get('GIT_COMMIT', 'dev')
            frame_q_size = video_source.frames_queue.maxsize
            audio_q_size = transcriber.audio_queue.maxsize
            logger.info(f"🚀 BUILD TAG: {git_commit} | frame_q={frame_q_size} | audio_q={audio_q_size}")
            logger.info(f"🔧 RUNTIME VERIFICATION: VideoSource.frames_queue.maxsize={frame_q_size}, TranscriptionEngine.audio_queue.maxsize={audio_q_size}")
            
            # Get video information
            width, height, fps = video_source.get_video_info()
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
            logger.info(f"[DEBUG] Video start time: {video_source.playback_start_time}")
            
            # Join the warm-up barrier (main thread is 3rd participant)
            logger.info("Main thread joining warm-up barrier...")
            try:
                video_source.warm_up_barrier.wait()
                logger.info("Main thread passed warm-up barrier")
            except threading.BrokenBarrierError:
                logger.error("Warm-up barrier was broken, falling back to event-based sync")
                video_source.warm_up_complete.set()

            # Set the wall-clock start time now that warm-up is complete
            if CLOCK.start_wall_time is None:
                CLOCK.start_wall_time = time.time()
                logger.info(f"🔧 Set singleton clock start_wall_time = {CLOCK.start_wall_time:.2f} (wall-clock time)")
            
            # Set compatibility property for existing code
            video_source.playback_start_time = CLOCK.start_wall_time
            logger.info(f"[SYNC] Using singleton clock: media_seek_pts={CLOCK.media_seek_pts:.2f}s, start_wall_time={CLOCK.start_wall_time:.2f}")

            # Now set the caption overlay video start time using the singleton clock
            caption_overlay.set_video_start_time(CLOCK.start_wall_time)
            logger.info(f"[SYNC] Caption overlay video start time set to: {CLOCK.start_wall_time}")

            # Set playback start time for correct timestamping in transcription
            transcriber.playback_start_time = video_source.playback_start_time
            logger.info(f"[SYNC] Set transcriber.playback_start_time = {transcriber.playback_start_time:.2f}")
            
            # Wait a brief moment after warm-up to ensure timing is stable
            time.sleep(0.5)
            
            # Get the current time after warm-up is complete
            current_playback_time = time.time() - video_source.playback_start_time
            logger.info("\n=== SCHEDULING TEST CAPTIONS ===")
            logger.info(f"System time: {time.time()}")
            logger.info(f"Video start time: {video_source.playback_start_time}")
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
            
            # Initialize and run the video processing loop with VideoRunner
            video_runner = VideoRunner(video_source, transcriber, caption_overlay)
            
            # Pre-buffer frames and start playback
            video_runner.prebuffer_frames()
            video_runner.run()
            
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
