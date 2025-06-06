#!/usr/bin/env python3
"""
Test to identify the exact problem with PIL text dimensions vs background sizing.
"""

import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from src.caption_overlay.renderer import CaptionRenderer
from PIL import Image, ImageDraw, ImageFont

def analyze_background_sizing_problem():
    """Analyze the exact problem with text extending beyond backgrounds."""
    
    renderer = CaptionRenderer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test with a specific problematic text
    test_text = "We're coming to now the end of the first six chapters."
    
    print('=' * 80)
    print('DETAILED FONT AND MEASUREMENT ANALYSIS')
    print('=' * 80)
    
    print(f'Test text: "{test_text}"')
    
    # Step 1: Manually test different measurement approaches
    # Get the same font that the renderer uses
    pil_font_size = max(16, int(renderer.style.font_scale * 30))
    pil_font = renderer._get_unicode_font(pil_font_size)
    
    print(f'Font size used: {pil_font_size}')
    print(f'Font object: {pil_font}')
    
    # Test different measurement methods
    bbox = pil_font.getbbox(test_text)
    width_bbox = bbox[2] - bbox[0]
    height_bbox = bbox[3] - bbox[1]
    
    length = pil_font.getlength(test_text)
    
    print(f'PIL getbbox: width={width_bbox}, height={height_bbox}')
    print(f'PIL getlength: {length}')
    print(f'With shadow offset: width={width_bbox + 2}, height={height_bbox + 2}')
    
    # Step 2: Test the renderer's calculation
    display_lines = renderer.process_caption_text(test_text)
    line_heights, line_widths, total_text_height, max_text_width = renderer.calculate_text_dimensions(display_lines)
    
    print(f'Renderer calculation: width={max_text_width}, height={total_text_height}')
    
    # Step 3: Test actual rendering and create a measurement image  
    test_image = Image.new('RGB', (800, 100), color=(0, 0, 0))
    draw = ImageDraw.Draw(test_image)
    
    # Render text the same way as the renderer
    shadow_offset = 2
    shadow_color = (0, 0, 0)
    text_color = (255, 255, 255)
    start_x = 10
    start_y = 10
    
    # Draw shadow and main text
    draw.text((start_x + shadow_offset, start_y + shadow_offset), test_text, font=pil_font, fill=shadow_color)
    draw.text((start_x, start_y), test_text, font=pil_font, fill=text_color)
    
    # Analyze the actual pixel bounds
    pixels = np.array(test_image)
    gray = pixels[:, :, 0] + pixels[:, :, 1] + pixels[:, :, 2]
    modified_mask = gray > 0
    
    if np.any(modified_mask):
        y_coords, x_coords = np.where(modified_mask)
        actual_left = np.min(x_coords)
        actual_right = np.max(x_coords)
        actual_width = actual_right - actual_left
        actual_top = np.min(y_coords)
        actual_bottom = np.max(y_coords)
        actual_height = actual_bottom - actual_top
        
        print(f'Actual pixel bounds: width={actual_width}, height={actual_height}')
        print(f'Actual span: x={actual_left} to {actual_right}, y={actual_top} to {actual_bottom}')
        
        # Calculate the effective text area (excluding the start position)
        effective_width = actual_width - start_x if actual_left <= start_x else actual_width
        print(f'Effective text width: {effective_width}')
    
    # Step 4: Test with the renderer's full caption rendering
    test_caption = {
        'text': test_text,
        'start_time': 0,
        'end_time': 5,
        'language': 'en'
    }
    
    frame_copy = frame.copy()
    result_frame = renderer.render_caption(frame_copy, test_caption, 2.5, 0, 'en')
    
    gray = result_frame[:, :, 0] + result_frame[:, :, 1] + result_frame[:, :, 2]
    modified_mask = gray > 0
    
    if np.any(modified_mask):
        y_coords, x_coords = np.where(modified_mask)
        actual_left = np.min(x_coords)
        actual_right = np.max(x_coords)
        actual_width = actual_right - actual_left
        
        print(f'Full caption render: width={actual_width}, bounds={actual_left} to {actual_right}')
    
    print('\n' + '=' * 80)

if __name__ == '__main__':
    analyze_background_sizing_problem() 