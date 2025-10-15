import time
import cv2
import os
import numpy as np

import cv2
import numpy as np

def auto_crop(image, threshold_ratio=0.05, min_crop=5):
    """
    Bonus 4: Automatic Edge Cropping
    Detects and removes black borders or alignment artifacts automatically
    by analyzing pixel intensity on all four edges (top, bottom, left, right).

    Returns:
        cropped (ndarray): Cropped version of the image
    """

    # Convert image to grayscale for intensity analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute intensity threshold (minimum 10 to handle bright images)
    threshold = max(10, np.mean(gray) * threshold_ratio)

    h, w = gray.shape

    # --- TOP EDGE ---
    top = 0
    for y in range(h // 3):  # Only check the upper third
        if np.mean(gray[y, :]) > threshold:
            top = max(y - min_crop, 0)
            break

    # --- BOTTOM EDGE ---
    bottom = h
    for y in range(h - 1, h // 3, -1):  # Scan upward from bottom
        if np.mean(gray[y, :]) > threshold:
            bottom = min(y + min_crop, h)
            break

    # --- LEFT EDGE ---
    left = 0
    for x in range(w // 3):  # Only check the left third
        if np.mean(gray[:, x]) > threshold:
            left = max(x - min_crop, 0)
            break

    # --- RIGHT EDGE ---
    right = w
    for x in range(w - 1, w // 3, -1):  # Scan leftward from right
        if np.mean(gray[:, x]) > threshold:
            right = min(x + min_crop, w)
            break

    # --- CROP IMAGE ---
    cropped = image[top:bottom, left:right]

    return cropped


def measure_time(func):
    """Decorator for measuring function execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        return result, duration
    return wrapper


def save_results(result_dir, base_name, unaligned, aligned, enhanced):
    """Saves output images to results directory."""
    os.makedirs(result_dir, exist_ok=True)
    cv2.imwrite(os.path.join(result_dir, f"{base_name}_unaligned.jpg"), unaligned)
    cv2.imwrite(os.path.join(result_dir, f"{base_name}_aligned.jpg"), aligned)
    cv2.imwrite(os.path.join(result_dir, f"{base_name}_enhanced.jpg"), enhanced)
