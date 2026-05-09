"""
Step 2 — Image Preprocessing
OpenCV pipeline: deskew → denoise → contrast enhance → adaptive threshold.
Poor preprocessing is the #1 cause of OCR errors on scanned documents.
"""
import cv2
import numpy as np
from PIL import Image

# ── Orientation Detection ──────────────────────────────────────────────────

def detect_and_fix_orientation(image_pil: Image.Image, orientation_predictor) -> Image.Image:
    """
    Detects if the page is rotated (90, 180, 270 degrees) and fixes it.
    Uses Surya's orientation predictor.
    """
    from surya.postprocessing.orientation import get_orientation
    
    # Run prediction
    results = orientation_predictor([image_pil])
    result = results[0]
    
    # Get the rotation angle (0, 90, 180, 270)
    angle = result.rotation_angle
    
    if angle != 0:
        print(f"[Preprocessor] Detected rotation: {angle} degrees. Fixing...")
        # PIL rotate is counter-clockwise, Surya angles are clockwise
        return image_pil.rotate(-angle, expand=True)
    
    return image_pil


# ── Individual steps ──────────────────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct rotation caused by uneven page placement during scanning.
    Uses minAreaRect on foreground pixels to find the skew angle.
    """
    # Invert so text is white on black (needed for contour detection)
    inverted = cv2.bitwise_not(image)
    coords   = np.column_stack(np.where(inverted > 0))

    if len(coords) < 5:          # Not enough points — skip deskew
        return image

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect angle is in [-90, 0); normalise to a small rotation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip if rotation is negligible (< 0.3°)
    if abs(angle) < 0.3:
        return image

    (h, w)  = image.shape[:2]
    center  = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Non-local means denoising — good for salt-and-pepper noise from old scans.
    h=10 is a balanced value; increase for heavily degraded pages.
    """
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Handles uneven lighting and faded ink across a scanned page.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding — produces a clean black/white image.
    Works better than Otsu for pages with uneven illumination.
    """
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )


# ── Full chain ────────────────────────────────────────────────────────────────

def preprocess(image_pil: Image.Image, orientation_predictor=None) -> np.ndarray:
    """
    Full preprocessing chain.

    Input:  PIL Image (any mode)
    Output: NumPy uint8 array — binarised grayscale ready for OCR
    """
    # 1. Orientation Fix (Upside down / Sideways)
    if orientation_predictor:
        image_pil = detect_and_fix_orientation(image_pil, orientation_predictor)

    # 2. Standard CV pipeline
    img = np.array(image_pil.convert("RGB"))   # ensure RGB
    img = to_grayscale(img)
    img = deskew(img)
    img = remove_noise(img)
    img = enhance_contrast(img)
    img = adaptive_threshold(img)
    return img


def preprocessed_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a preprocessed ndarray back to a PIL Image (for OCR engines that need PIL)."""
    return Image.fromarray(arr).convert("RGB")
