import os
import cv2
import numpy as np
from alignment import split_image, align_channels, pyramid_align
from enhancement import enhance_image
from utils import save_results, measure_time, auto_crop

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/images")
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "../results")

# Variables to store the state of bonus features
USE_PYRAMID = False
USE_AUTOCROP = False
CURRENT_RESULTS_DIR = DEFAULT_RESULTS_DIR 

@measure_time
def process_image(path, use_pyramid, use_autocrop, results_dir):
    """
    Loads an image, aligns its channels, applies optional cropping,
    enhances it, and saves the results.
    """
    # Load image as grayscale (single channel)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image could not be loaded: {path}")
    
    print(f"→ Processing {os.path.basename(path)}")
    print(f"   Pyramid Alignment: {'ON' if use_pyramid else 'OFF'}")
    print(f"   Auto Crop: {'ON' if use_autocrop else 'OFF'}")


    # Split the stacked grayscale image into B, G, R channels
    B, G, R = split_image(img)

    # === Channel Alignment ===
    # Reference channel is B (Blue)
    if use_pyramid:
        # Use Pyramid Alignment (Bonus 5)
        shift_G = pyramid_align(B, G)
        shift_R = pyramid_align(B, R)
    else:
        # Use Standard Alignment (SSD metric)
        shift_G = align_channels(B, G, 15, metric="ssd")
        shift_R = align_channels(B, R, 15, metric="ssd")

    # Apply the calculated shifts
    G_aligned = np.roll(G, shift_G, axis=(0, 1))
    R_aligned = np.roll(R, shift_R, axis=(0, 1))

    # Merge channels to create BGR images
    unaligned = cv2.merge([B, G, R])
    aligned = cv2.merge([B, G_aligned, R_aligned])

    # === Automatic Cropping ===
    if use_autocrop:
        # Apply Automatic Edge Cropping (Bonus 4)
        aligned = auto_crop(aligned)

    # === Image Enhancement ===
    enhanced = enhance_image(aligned)

    base_name = os.path.splitext(os.path.basename(path))[0]
    
    # Save output images
    save_results(results_dir, base_name, unaligned, aligned, enhanced)

    return shift_G, shift_R


if __name__ == "__main__":
    
    # --- User Input for Bonus Features ---
    
    # Query for Pyramid Alignment
    pyramid_input = input("Enable Pyramid Alignment (Bonus 5)? (y/n): ").strip().lower()
    USE_PYRAMID = pyramid_input == 'y'
    
    # Query for Automatic Edge Cropping
    crop_input = input("Enable Automatic Edge Cropping (Bonus 4)? (y/n): ").strip().lower()
    USE_AUTOCROP = crop_input == 'y'
    
    # --- Determine Results Directory ---
    
    if USE_PYRAMID or USE_AUTOCROP:
        # If at least one bonus is enabled, use a specific directory name
        bonus_parts = []
        if USE_PYRAMID:
            bonus_parts.append("Pyramid")
        if USE_AUTOCROP:
            bonus_parts.append("Crop")
        
        bonus_suffix = "_with_" + "_".join(bonus_parts)
        # Create a new results directory path
        CURRENT_RESULTS_DIR = DEFAULT_RESULTS_DIR.replace("results", "results" + bonus_suffix)
    else:
        # If no bonus is enabled, use a 'no_bonus' directory
        CURRENT_RESULTS_DIR = os.path.join(BASE_DIR, "../results")

    # Create the results directory if it doesn't exist
    if not os.path.exists(CURRENT_RESULTS_DIR):
        os.makedirs(CURRENT_RESULTS_DIR)

    print(f"\nResults will be saved to: {CURRENT_RESULTS_DIR}")
    print(f"Pyramid Alignment: {'ENABLED' if USE_PYRAMID else 'DISABLED'}")
    print(f"Automatic Cropping: {'ENABLED' if USE_AUTOCROP else 'DISABLED'}\n")

    # --- Process all images in the data directory ---
    for file in os.listdir(DATA_DIR):
        if file.endswith((".jpg", ".tif", ".png")): # Added common image extensions
            path = os.path.join(DATA_DIR, file)
            
            # Process the image with current bonus settings
            (shift_G, shift_R), duration = process_image(
                path, 
                USE_PYRAMID, 
                USE_AUTOCROP, 
                CURRENT_RESULTS_DIR
            )
            print(f"Processed {file}: G{shift_G}, R{shift_R} | Time: {duration:.2f}s")