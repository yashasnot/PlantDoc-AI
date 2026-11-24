
import os
import sys
import cv2

# Get the project root directory
ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Add project root to Python import path
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.reliability.image_quality import assess_image_quality

image_path = os.path.join(
    ROOT,
    "outputs",
    "predictions",
    "sample_isolated_leaf.png"
)

mask_path = os.path.join(
    ROOT,
    "outputs",
    "masks",
    "sample_leaf_mask.png"
)

image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(image_path)

if mask is None:
    raise FileNotFoundError(mask_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = assess_image_quality(image, mask)

print("=" * 45)
print("LEAF-AWARE IMAGE QUALITY REPORT")
print("=" * 45)

for key, value in result.items():
    print(f"{key}: {value}")
