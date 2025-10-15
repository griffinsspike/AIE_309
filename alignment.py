import numpy as np
import cv2

def split_image(image):
    """Splits grayscale image into B, G, R channels (top to bottom)."""
    h = image.shape[0] // 3
    B = image[0:h, :]
    G = image[h:2*h, :]
    R = image[2*h:3*h, :]
    return B, G, R


def ssd_metric(img1, img2):
    """Computes Sum of Squared Differences (SSD) between two images."""
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.sum(diff ** 2)


def ncc_metric(img1, img2):
    """Computes Normalized Cross-Correlation (NCC) between two images."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1_mean, img2_mean = np.mean(img1), np.mean(img2)

    numerator = np.sum((img1 - img1_mean) * (img2 - img2_mean))
    denominator = np.sqrt(np.sum((img1 - img1_mean)**2) * np.sum((img2 - img2_mean)**2) + 1e-8)
    return numerator / denominator


def align_channels(ref, target, search_range=15, metric="ssd"):
    """
    Aligns target channel to reference channel using SSD or NCC.
    Returns the optimal (dy, dx) shift.
    """
    best_score = float('inf') if metric == "ssd" else -float('inf')
    best_shift = (0, 0)

    h, w = ref.shape
    crop_h = int(0.1 * h)
    crop_w = int(0.1 * w)

    # Crop both images to ignore noisy borders (10%)
    ref_cropped = ref[crop_h:-crop_h, crop_w:-crop_w]

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            # Kaydırma: dışarı taşan kısımlar 0 olacak (wrap-around yok)
            shifted = np.roll(target, (dy, dx), axis=(0, 1))
            shifted[:crop_h, :] = 0
            shifted[-crop_h:, :] = 0
            shifted[:, :crop_w] = 0
            shifted[:, -crop_w:] = 0

            shifted_cropped = shifted[crop_h:-crop_h, crop_w:-crop_w]

            # Hesapla
            score = (
                ssd_metric(ref_cropped, shifted_cropped)
                if metric == "ssd"
                else ncc_metric(ref_cropped, shifted_cropped)
            )

            if (metric == "ssd" and score < best_score) or (metric == "ncc" and score > best_score):
                best_score = score
                best_shift = (dy, dx)

    return best_shift


def pyramid_align(ref, target, levels=3, metric="ssd"):
    """
    Pyramid-based alignment for large images.
    Returns (dy, dx) shift with coarse-to-fine strategy.
    """
    # Base case: lowest pyramid level
    if levels == 0 or min(ref.shape) < 50:
        return align_channels(ref, target, search_range=15, metric=metric)

    # Create downsampled versions
    ref_small = cv2.pyrDown(ref)
    target_small = cv2.pyrDown(target)

    # Recursive alignment at smaller scale
    dy, dx = pyramid_align(ref_small, target_small, levels - 1, metric)

    # Scale back the shift
    dy, dx = dy * 2, dx * 2

    # Refine around the scaled estimate
    refined_shift = align_channels(
        ref, np.roll(target, (dy, dx), axis=(0, 1)), search_range=2, metric=metric
    )
    final_dy = dy + refined_shift[0]
    final_dx = dx + refined_shift[1]

    return (final_dy, final_dx)
