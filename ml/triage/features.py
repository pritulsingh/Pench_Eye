import cv2
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class ImageFeatures:
    brightness: float          # mean pixel value normalized 0-1
    contrast: float            # std of pixel values normalized 0-1
    entropy: float             # image entropy (Shannon) normalized 0-1
    blur_score: float          # Laplacian variance normalized 0-1 (higher = sharper)
    saturation: float          # mean saturation in HSV space, 0-1
    edge_density: float        # Canny edge pixel ratio, 0-1
    dark_pixel_ratio: float    # ratio of pixels < 15 intensity
    bright_pixel_ratio: float  # ratio of pixels > 240 intensity
    resolution_score: float    # 1.0 if >= 640x480, scales down
    quality_score: float       # composite score (weighted average)

class ImageFeatureExtractor:
    """Extracts low-level image features for quality assessment and blank classification."""
    
    def extract(self, image: np.ndarray) -> ImageFeatures:
        """Extract features from the image."""
        if image is None or image.size == 0:
            return ImageFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        h, w = image.shape[:2]
        total_pixels = h * w
        if total_pixels == 0:
            return ImageFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = np.clip(np.mean(hsv[:, :, 1]) / 255.0, 0.0, 1.0)
        else:
            gray = image
            saturation = 0.0
            
        # Brightness and contrast
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        brightness = np.clip(mean_val / 255.0, 0.0, 1.0)
        contrast = np.clip(std_val / 128.0, 0.0, 1.0) # normalize std roughly
        
        # Entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / total_pixels
        non_zero_hist = hist[hist > 0]
        entropy_val = -np.sum(non_zero_hist * np.log2(non_zero_hist))
        entropy = np.clip(entropy_val / 8.0, 0.0, 1.0) # max entropy is 8 for 8-bit image
        
        # Blur score
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = np.clip(laplacian_var / 1000.0, 0.0, 1.0) # roughly normalized
        
        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.clip(np.count_nonzero(edges) / total_pixels, 0.0, 1.0)
        
        # Dark / Bright pixel ratios
        dark_pixel_ratio = np.clip(np.count_nonzero(gray < 15) / total_pixels, 0.0, 1.0)
        bright_pixel_ratio = np.clip(np.count_nonzero(gray > 240) / total_pixels, 0.0, 1.0)
        
        # Resolution score
        target_pixels = 640 * 480
        resolution_score = np.clip(total_pixels / target_pixels, 0.0, 1.0)
        
        # Quality score
        quality_score = (
            0.25 * blur_score +
            0.20 * contrast +
            0.15 * entropy +
            0.15 * (1.0 - dark_pixel_ratio) +
            0.15 * (1.0 - bright_pixel_ratio) +
            0.10 * saturation +
            0.10 * resolution_score
        )
        quality_score = float(np.clip(quality_score, 0.0, 1.0))
        
        return ImageFeatures(
            brightness=float(brightness),
            contrast=float(contrast),
            entropy=float(entropy),
            blur_score=float(blur_score),
            saturation=float(saturation),
            edge_density=float(edge_density),
            dark_pixel_ratio=float(dark_pixel_ratio),
            bright_pixel_ratio=float(bright_pixel_ratio),
            resolution_score=float(resolution_score),
            quality_score=quality_score
        )
