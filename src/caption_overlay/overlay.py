"""Orchestration logic for caption overlay operations."""
import time
import logging

# Import with try-except to handle both direct execution and module import
try:
    from ..logging_utils import get_logger, TRACE
    from .core import CaptionCore
    from .renderer import CaptionRenderer
    from .style_config import CaptionStyleConfig
except ImportError:
    from src.logging_utils import get_logger, TRACE
    from .core import CaptionCore
    from .renderer import CaptionRenderer
    from .style_config import CaptionStyleConfig

logger = get_logger(__name__)

class CaptionOverlayOrchestrator:
    """Orchestrates caption overlay operations between core state and rendering."""
    
    def __init__(self, core=None, renderer=None, style_config=None):
        """Initialize the caption overlay orchestrator.
        
        Args:
            core: CaptionCore instance or None for default
            renderer: CaptionRenderer instance or None for default
            style_config: CaptionStyleConfig instance or None for defaults
        """
        self.core = core if core is not None else CaptionCore()
        self.renderer = renderer if renderer is not None else CaptionRenderer(style_config)
        logger.debug("CaptionOverlayOrchestrator initialized")
    
    def overlay_captions(self, frame, current_time=None, frame_count=0):
        """Overlay all valid captions on frame.
        
        Args:
            frame: The frame to overlay captions on
            current_time: The current timestamp in seconds (relative to video start)
            frame_count: The current frame number (used for logging)
            
        Returns:
            Frame with captions drawn
        """
        # Start timing the rendering operation
        render_start = time.time()
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        if current_time is None:
            logger.warning("[OVERLAY] No current_time provided, skipping caption overlay")
            return frame
            
        # Ensure current_time is relative to video start and non-negative
        current_relative_time = max(0, current_time)
        
        # Log timing info less frequently for production
        if frame_count % 180 == 0:  # Log every 6 seconds at 30fps
            logger.debug(f"[OVERLAY] Frame {frame_count} | Time: {current_relative_time:.2f}s | Captions in queue: {self.core.get_caption_count()}")
            if self.core.get_caption_count() > 0 and logger.isEnabledFor(TRACE):
                logger.trace("[OVERLAY] Next caption in queue:")
                # Get upcoming captions for logging
                sorted_captions = sorted(self.core.captions, key=lambda x: x['start_time'])
                for i, c in enumerate(sorted_captions[:3]):
                    time_until = c['start_time'] - current_relative_time
                    status = "ACTIVE " if c['start_time'] <= current_relative_time <= c['end_time'] else "PENDING"
                    logger.trace(f"  {i+1}. [{status}] In {time_until:6.2f}s: '{c['text'][:50]}{'...' if len(c['text']) > 50 else ''}'")
        
        # Log caption state less frequently for production
        log_frequency = 90  # Log every 3 seconds at 30fps
        should_log = frame_count % log_frequency == 0
        
        if should_log and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[OVERLAY] Processing frame {frame_count} at {current_relative_time:.2f}s")
            
            if logger.isEnabledFor(TRACE):
                logger.trace(
                    f"[TIMING] Frame {frame_count} details - "
                    f"Relative time: {current_relative_time:.6f}s | "
                    f"Video start: {self.core.video_start_time:.6f} | "
                    f"System time: {time.time():.6f} | "
                    f"Time since start: {time.time() - self.core.video_start_time:.6f}s"
                )
        
        # Get active captions from core
        active_captions = self.core.get_active_captions(current_relative_time)
        
        # If no captions, return the frame as-is
        if not active_captions:
            if should_log:
                # Move queue status to TRACE level
                logger.trace(f"[OVERLAY] No active captions at {current_relative_time:.3f}s")
                if self.core.get_caption_count() > 0:
                    upcoming = [c for c in sorted(self.core.captions, key=lambda x: x['start_time']) 
                              if c['start_time'] > current_relative_time]
                    if upcoming:
                        logger.trace("[OVERLAY] Upcoming captions:")
                        for c in upcoming[:3]:  # Show next 3 captions
                            time_until = c['start_time'] - current_relative_time
                            logger.trace(
                                f"  - In {time_until:6.3f}s: "
                                f"'{c['text'][:50]}{'...' if len(c['text']) > 50 else ''}'"
                            )
                        if len(upcoming) > 3:
                            logger.trace(f"  - ... and {len(upcoming) - 3} more captions")
            return frame
        
        if should_log:
            logger.info(f"[OVERLAY] Found {len(active_captions)} active captions")
            for i, cap in enumerate(active_captions, 1):
                logger.info(f"  {i}. '{cap['text']}' ({cap['start_time']:.2f}-{cap['end_time']:.2f}s)")
        
        # Log caption timing info for debugging
        if should_log and self.core.get_caption_count() > 0:
            logger.trace("[OVERLAY] Caption timing details:")
            for i, c in enumerate(sorted(self.core.captions, key=lambda x: x['start_time'])):
                time_until = c['start_time'] - current_relative_time
                time_remaining = c['end_time'] - current_relative_time
                
                # Determine status and time info
                if time_until > 0:
                    status = "PENDING"
                    time_info = f"starts in {time_until:.1f}s"
                elif time_remaining < 0:
                    status = "ENDED  "
                    time_info = f"ended {abs(time_remaining):.1f}s ago"
                else:
                    status = "ACTIVE "
                    time_info = f"active, {time_remaining:.1f}s remaining"
                
                # Log caption info
                logger.info(
                    f"  {i+1:2d}. [{status}] {time_info:>25s} | "
                    f"'{c['text'][:40]}{'...' if len(c['text']) > 40 else ''}'"
                )
                
                # Log timing details
                logger.debug(
                    f"      Start: {c['start_time']:.2f}s | "
                    f"End: {c['end_time']:.2f}s | "
                    f"Duration: {c['end_time']-c['start_time']:.2f}s"
                    f"{' | Added: ' + str(round(time.time() - c['added_at'], 1)) + 's ago' if 'added_at' in c else ''}"
                )
                
                # Log timing mode if available
                if 'was_absolute' in c:
                    abs_status = "ABSOLUTE" if c['was_absolute'] else "RELATIVE"
                    logger.debug(f"      Timing: {abs_status} | Original: {c.get('original_timestamp', 'N/A')}")
        
        # Render all active captions using the renderer
        result_frame = self.renderer.render_multiple_captions(frame, active_captions, current_relative_time)
        
        # Debug logging for rendered captions
        if active_captions:
            display_lines = []
            for caption in active_captions:
                lines = self.renderer.process_caption_text(caption['text'])
                display_lines.extend(lines)
            logger.debug(f"Rendered captions at {current_time:.2f}s: {display_lines}")
        
        overlay_duration = time.time() - render_start
        if frame_count % 30 == 0: # Log overlay duration periodically (e.g., every second at 30fps)
            logger.debug(f"[TIMING] CaptionOverlayOrchestrator.overlay_captions took {overlay_duration*1000:.2f}ms for frame {frame_count}")

        return result_frame
    
    # Proxy methods to core functionality for backward compatibility
    def add_caption(self, text, timestamp, duration=1.0, is_absolute=False, seamless=True, language='en', is_primary=True):
        """Add a caption to be displayed. Proxy to core.add_caption."""
        # Pass all arguments to CaptionCore.add_caption, including new language and is_primary parameters
        return self.core.add_caption(text, timestamp, duration, is_absolute, language, is_primary)
    
    def set_video_start_time(self, start_time):
        """Set the video's start time. Proxy to core.set_video_start_time."""
        return self.core.set_video_start_time(start_time)
    
    def prune_captions(self, current_time, buffer=1.0):
        """Prune old captions. Proxy to core.prune_captions."""
        return self.core.prune_captions(current_time, buffer)
    
    @property
    def captions(self):
        """Access to the captions list for backward compatibility."""
        return self.core.captions
    
    @property
    def video_start_time(self):
        """Access to video start time for backward compatibility."""
        return self.core.video_start_time
    
    @property
    def lock(self):
        """Access to the threading lock for backward compatibility."""
        return self.core.lock 