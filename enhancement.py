import cv2
import numpy as np

def enhance_image(image):
    """
    Enhances the aligned image using two techniques
    1. Histogram Equalization – improves contrast
    2. Gamma Correction – adjusts brightness
    """

    # Histogram Equalization 
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    hist_eq = cv2.merge(eq_channels)

    # Gamma Correction
    gamma = 1.2   
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(hist_eq, table)

    return gamma_corrected

