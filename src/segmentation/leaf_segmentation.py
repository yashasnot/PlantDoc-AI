
import cv2
import numpy as np

def segment_leaf(image):

    image = np.asarray(image.convert("RGB"))

    h, w = image.shape[:2]

    scale = min(1.0, 800 / max(h, w))

    if scale < 1:
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale))
        )

    h, w = image.shape[:2]

    bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    mask = np.zeros((h, w), np.uint8)

    rect = (
        int(w * .04),
        int(h * .04),
        int(w * .92),
        int(h * .92)
    )

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        bgr, mask, rect, bgd, fgd,
        5, cv2.GC_INIT_WITH_RECT
    )

    result = np.where(
        (mask == 1) | (mask == 3),
        255, 0
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    result = cv2.morphologyEx(
        result, cv2.MORPH_CLOSE, kernel
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            result, 8
        )
    )

    if count > 1:
        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
        result = np.where(
            labels == largest, 255, 0
        ).astype(np.uint8)

    isolated = cv2.bitwise_and(
        image, image, mask=result
    )

    return result, isolated
