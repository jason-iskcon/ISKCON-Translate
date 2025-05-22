# Enhanced Logging System Implementation Plan

## Development Approach

### Core Principles
1. **Minimal Viable Changes (MVC)**
   - Make the smallest possible change that moves us forward
   - Verify each change before proceeding
   - Avoid scope creep in individual changes

2. **Incremental Development**
   - Work in small, testable increments
   - Each change should be independently verifiable
   - Maintain a working version at all times

3. **Testing Strategy**
   - Manual verification after each significant change
   - Document test cases before implementation
   - Plan to add automated tests in future iterations

4. **Debugging Protocol**
   - If debugging takes >15 minutes, revert and try a different approach
   - Keep detailed notes of issues and solutions
   - Maintain a known-good version to compare against

5. **Documentation**
   - Update documentation with each change
   - Keep track of decisions and their rationale
   - Document any workarounds or temporary solutions

## Implementation Plan

This document outlines the step-by-step process to implement an enhanced logging system with three distinct log levels (INFO, DEBUG, TRACE) for better diagnostics and debugging.

This document outlines the step-by-step process to implement an enhanced logging system with three distinct log levels (INFO, DEBUG, TRACE) for better diagnostics and debugging.

## Phase 1: Setup and Infrastructure

- [x] **1.1 Create logging_utils.py**
  - [x] Add TRACE log level definition
  - [x] Implement trace() method for logger
  - [x] Create helper functions for consistent log formatting
  - [x] Add colored output for better log readability
  - [x] Implement setup_logging() and get_logger() utilities

- [x] **1.2 Update main.py**
  - [x] Import and initialize the new logging utilities
  - [x] Set up log level configuration from command line arguments
  - [x] Add --log-level flag with choices: INFO, DEBUG, TRACE
  - [x] Update logging calls to use new format
  - [x] Add command-line argument parsing for input file and log level

- [x] **1.3 Verification: Basic Logging**
  - [x] Verify log levels work (INFO, DEBUG, TRACE)
  - [x] Check log output format and colors
  - [x] Test thread safety
  - [x] Verify log file creation
  - [x] Check performance impact (minimal overhead observed) in production mode (INFO)

## Phase 2: Update Video Source

- [x] **2.1 Update video_source.py**
  - [x] Import TRACE level from logging_utils
  - [x] Categorize frame processing logs
  - [x] Move frame timing details to TRACE
  - [x] Keep video initialization in INFO
  - [x] Add frame counter logging at DEBUG level
  - [x] Log video properties at startup (resolution, FPS, etc.)

- [x] **2.2 Verification: Video Loading**
  - [x] Verify video loads correctly with new logging
  - [x] Check frame counter accuracy
  - [x] Validate video properties logging
  - [x] Test with different video formats (MP4, AVI, MKV)
  - [x] Measure performance impact of new logging
  - [x] Implement frame queue management
  - [x] Add detailed frame timing information
  - [x] Verify timestamp consistency
  - [x] Test with various resolutions and frame rates

- [ ] **2.3 Essential Error Handling**
  - [ ] **File Operations**
    - [ ] Handle missing video files with clear error messages
    - [ ] Detect and report corrupted video files
    - [ ] Verify file permissions before processing
  - [ ] **Video Format Support**
    - [ ] Verify support for common ISKCON formats (MP4/H.264)
    - [ ] Add clear error messages for unsupported formats
    - [ ] Test with typical ISKCON video resolutions (720p, 1080p)
  - [ ] **Resource Management**
    - [ ] Add basic memory monitoring for large files
    - [ ] Handle out-of-memory conditions gracefully
    - [ ] Add cleanup for temporary files
  - [ ] **Recovery**
    - [ ] Handle video source disconnections
    - [ ] Add automatic retry for temporary failures
    - [ ] Implement configurable timeouts

## Phase 3: Update Transcription Module

- [x] **3.1 Update transcription.py**
  - [x] Import TRACE level from logging_utils
  - [x] Categorize transcription logs:
    - INFO: Transcription start/end, major chunks
    - [x] DEBUG: Chunk processing decisions
    - [x] TRACE: Audio buffer details, timing calculations

- [x] **3.2 Update audio processing logs**
  - [x] Move detailed audio chunk analysis to TRACE
  - [x] Keep transcription results in INFO
  - [x] Move buffer management to DEBUG
  - [x] Add audio format and sample rate logging
  - [x] Log transcription confidence scores

- [ ] **3.3 Verification: Audio Processing**
  - [x] Verify transcription starts/stops correctly
    - [x] Check logs for proper initialization messages
    - [x] Verify worker thread starts and stops cleanly
    - [x] Confirm queues are properly cleared on stop
  - [x] Check audio format detection
    - [x] Verify format is logged correctly for different input types
    - [x] Add input validation for audio segments
    - [x] Test with invalid input formats
  - [x] Validate transcription accuracy logging
    - [x] Check confidence scores are logged when available
    - [x] Verify timestamps are accurate
    - [x] Add tests for timestamp and confidence score handling
  - [x] Test with different audio qualities - **COMPLETED**
    - [x] High quality (16-bit, 44.1kHz)
    - [x] Medium quality (16-bit, 22.05kHz)
    - [x] Low quality (8-bit, 8kHz)
    - [x] Verified Whisper handles various audio qualities effectively
    - [x] No significant impact on transcription accuracy observed
  - [x] Measure transcription performance impact - **COMPLETED**
    - [x] Log processing time per chunk
    - [x] Track memory usage during transcription
    - [x] Monitor queue sizes to prevent overflow
    - [x] Add CPU usage monitoring
    - [x] Suppress third-party debug logs for cleaner output
    - [x] Add queue full warnings at 80% capacity

## Phase 4: Update Caption Overlay (Final Phase - Handle with Care)

- [ ] **4.1 Add basic logging to caption_overlay.py**
  - [ ] Import TRACE level from logging_utils
  - [ ] Add initialization logging (font loading, settings)
  - [ ] Log caption addition/removal events
  - [ ] Keep minimal impact on rendering performance

- [ ] **4.2 Add debug visualization options**
  - [ ] Add frame counter overlay
  - [ ] Add timing information overlay
  - [ ] Make visualization togglable via debug flag
  - [ ] Add bounding box visualization for text placement

- [ ] **4.3 Add performance monitoring**
  - [ ] Log rendering time per frame
  - [ ] Track caption queue size
  - [ ] Monitor memory usage for leaks
  - [ ] Add warning for slow frame processing

- [ ] **4.4 Verification: Caption Rendering**
  - [ ] Verify captions appear correctly
  - [ ] Check timing synchronization
  - [ ] Validate debug visualizations
  - [ ] Test with different caption lengths
  - [ ] Measure rendering performance impact

## Phase 5: System-wide Testing and Validation

- [ ] **5.1 System Integration Test**
  - [ ] Test all components together
  - [ ] Verify log consistency across modules
  - [ ] Check system resource usage
  - [ ] Test error handling and recovery
  - [ ] Validate log file rotation and management

- [ ] **5.2 Performance Testing**
  - [ ] Measure impact of logging on CPU usage
  - [ ] Check memory usage patterns
  - [ ] Test with long-running sessions
  - [ ] Verify no memory leaks
  - [ ] Compare performance with/without TRACE logging


## Phase 6: Final Documentation and Handover

- [ ] **6.1 Update README.md**
  - Document the new logging system
  - Add examples of log levels and usage
  - Include command-line usage with --log-level

- [ ] **6.2 Add logging guidelines**
  - When to use each log level
  - Best practices for log messages
  - How to interpret logs for debugging

## Implementation Notes

1. **Log Message Format**
   - Use consistent prefixes like [CAPTION], [AUDIO], [VIDEO]
   - Include relevant context (timestamps, queue sizes, etc.)
   - Keep messages concise but informative

2. **Performance Considerations**
   - Use lazy evaluation for expensive log operations
   - Structure log messages to minimize string formatting overhead
   - Consider conditional logging for frequently executed code paths

3. **Error Handling**
   - Log errors with stack traces at ERROR level
   - Include recovery information when available
   - Use structured logging for better parsing

## Example Command Line Usage

```bash
# Production mode (default: INFO)
python src/main.py --input video.mp4

# Development mode
python src/main.py --input video.mp4 --log-level DEBUG

# Deep debugging
python src/main.py --input video.mp4 --log-level TRACE
```

## Completion Checklist

- [ ] All log statements reviewed and categorized
- [ ] Log levels tested at each level
- [ ] Documentation updated
- [ ] Performance impact verified
- [ ] Code reviewed and merged
