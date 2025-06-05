from src.caption_overlay import CaptionOverlay
import time

print("Creating overlay...")
overlay = CaptionOverlay()
overlay.set_video_start_time(time.time())

print("Adding caption...")
result = overlay.add_caption('Test Caption', 0.0, 2.0)
print(f"Add result: {result}")

print(f"Total captions: {len(overlay.captions)}")
if overlay.captions:
    cap = overlay.captions[0]
    print(f"Caption details: {cap['text']} ({cap['start_time']:.1f}s-{cap['end_time']:.1f}s)")

print("Testing active captions via core...")
test_times = [0.0, 1.0, 2.0]
for t in test_times:
    try:
        active = overlay.core.get_active_captions(t)
        print(f"Time {t:.1f}s: {len(active)} active captions")
        for cap in active:
            print(f"  - '{cap['text']}' fade={cap.get('fade_factor', 'N/A')}")
    except Exception as e:
        print(f"Error at time {t}: {e}")
        import traceback
        traceback.print_exc()

print("Testing overlay_captions method...")
import numpy as np
frame = np.zeros((480, 640, 3), dtype=np.uint8)
for t in test_times:
    try:
        result_frame = overlay.overlay_captions(frame.copy(), current_time=t)
        is_different = not np.array_equal(frame, result_frame)
        print(f"Time {t:.1f}s: Frame modified = {is_different}")
    except Exception as e:
        print(f"Error rendering at time {t}: {e}")
        import traceback
        traceback.print_exc()

print("Done.") 