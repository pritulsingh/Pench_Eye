#!/usr/bin/env python3
"""
SAMPLE IMAGE GENERATOR — Tiger Intelligence System
Generates synthetic camera-trap images for hackathon demonstration.
All images are SYNTHETIC — not real wildlife photographs.
"""

import os
import random
from datetime import datetime, timedelta

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Error: Pillow is not installed. Run: pip install Pillow")
    import sys
    sys.exit(1)

def generate_images():
    base_dir = "./sample_data/batch_001"
    
    # Create directories for CAM-001 to CAM-004
    for i in range(1, 5):
        os.makedirs(os.path.join(base_dir, f"CAM-{i:03d}"), exist_ok=True)
        
    print("Generating 100 synthetic test images...")
    
    manifest = {"blank": 0, "subject": 0}
    
    for i in range(1, 101):
        img = Image.new('RGB', (800, 600))
        draw = ImageDraw.Draw(img)
        
        is_blank = i <= 32  # 32 blank images
        
        if is_blank:
            # Generate blank-like images (dark, blurry, overexposed)
            bg_color = random.choice([(10, 10, 10), (250, 250, 250), (50, 50, 50)])
            draw.rectangle([0, 0, 800, 600], fill=bg_color)
            manifest["blank"] += 1
        else:
            # Generate subject-like images (foliage/animal shapes)
            bg_color = (34, 139, 34) # Forest green base
            draw.rectangle([0, 0, 800, 600], fill=bg_color)
            
            # Draw some shapes to represent a tiger
            draw.ellipse([300, 200, 500, 400], fill=(210, 105, 30)) # Orange body
            # Add stripes
            for _ in range(5):
                x = random.randint(320, 480)
                y = random.randint(220, 380)
                draw.line([(x, y), (x+10, y+50)], fill=(0, 0, 0), width=5)
            manifest["subject"] += 1
            
        # Assign to a random camera
        cam = f"CAM-{random.randint(1, 4):03d}"
        filename = f"IMG_{i:05d}.jpg"
        filepath = os.path.join(base_dir, cam, filename)
        
        # Save image
        img.save(filepath, "JPEG")
        
    print(f"Generated {manifest['blank']} blank images and {manifest['subject']} subject images.")
    print(f"Images saved to {base_dir}")

if __name__ == "__main__":
    generate_images()
