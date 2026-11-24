
import cv2
import numpy as np

def detect_lesions(image, leaf_mask):

    rgb = np.asarray(image.convert("RGB"))

    h, w = rgb.shape[:2]

    leaf_mask = cv2.resize(
        leaf_mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV
    )

    H, S, V = cv2.split(hsv)

    yellow = (
        (H >= 15) & (H <= 38) &
        (S >= 35) & (V >= 65)
    )

    brown = (
        (H <= 25) &
        (S >= 40) &
        (V >= 35) & (V <= 175)
    )

    dark = (
        (V <= 65) & (S >= 25)
    )

    candidates = (
        yellow | brown | dark
    ).astype(np.uint8) * 255

    lesion_mask = cv2.bitwise_and(
        candidates,
        candidates,
        mask=leaf_mask
    )

    kernel = np.ones((3, 3), np.uint8)

    lesion_mask = cv2.morphologyEx(
        lesion_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    lesion_mask = cv2.morphologyEx(
        lesion_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    leaf_area = np.count_nonzero(leaf_mask)
    lesion_area = np.count_nonzero(lesion_mask)

    severity = (
        100 * lesion_area / leaf_area
        if leaf_area else 0
    )

    overlay = rgb.copy()
    overlay[lesion_mask > 0] = (255, 0, 0)

    overlay = cv2.addWeighted(
        rgb, 0.65, overlay, 0.35, 0
    )

    return lesion_mask, overlay, severity
