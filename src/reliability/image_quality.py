
import cv2
import numpy as np


def assess_image_quality(image, leaf_mask=None):
    """
    Evaluate image quality.

    If a leaf mask is provided, calculate brightness,
    contrast and blur only within the leaf region.
    """

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image provided.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image.")

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Prepare optional leaf mask
    if leaf_mask is not None:
        if leaf_mask.shape[:2] != (height, width):
            leaf_mask = cv2.resize(
                leaf_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        if leaf_mask.ndim == 3:
            leaf_mask = cv2.cvtColor(
                leaf_mask, cv2.COLOR_RGB2GRAY
            )

        valid_pixels = leaf_mask > 0

        if np.count_nonzero(valid_pixels) < 100:
            raise ValueError("Leaf mask contains insufficient pixels.")
    else:
        valid_pixels = np.ones(
            (height, width), dtype=bool
        )

    # Extract pixels within the leaf region
    leaf_pixels = gray[valid_pixels]

    # Brightness and contrast
    brightness = float(np.mean(leaf_pixels))
    contrast = float(np.std(leaf_pixels))

    # Blur score
    laplacian = cv2.Laplacian(
        gray, cv2.CV_64F
    )

    blur_score = float(
        np.var(laplacian[valid_pixels])
    )

    total_pixels = height * width
    leaf_pixel_count = int(np.count_nonzero(valid_pixels))

    warnings = []

    if blur_score < 50:
        warnings.append("Image may be blurry.")

    if brightness < 50:
        warnings.append("Leaf region may be too dark.")

    if brightness > 210:
        warnings.append("Leaf region may be overexposed.")

    if contrast < 25:
        warnings.append("Leaf region has low contrast.")

    if height < 128 or width < 128:
        warnings.append("Image resolution is too low.")

    return {
        "image_width": width,
        "image_height": height,
        "total_pixels": total_pixels,
        "leaf_pixels": leaf_pixel_count,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "is_suitable": len(warnings) == 0,
        "quality_status": (
            "Suitable" if not warnings else "Needs Review"
        ),
        "warnings": warnings
    }
