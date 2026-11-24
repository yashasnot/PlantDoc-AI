
import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, ROOT)

from src.reliability.gate import (
    create_image_variants,
    DEFAULT_TEMPERATURE,
    CONFIDENCE_THRESHOLD,
    STABILITY_THRESHOLD
)

print("=" * 60)
print("PLANTDOC AI - RELIABILITY GATE MODULE TEST")
print("=" * 60)

print("Temperature:", DEFAULT_TEMPERATURE)
print("Confidence threshold:", CONFIDENCE_THRESHOLD)
print("Stability threshold:", STABILITY_THRESHOLD)

print("\nTesting image variants...")

from PIL import Image

sample = Image.new(
    "RGB",
    (256, 256),
    color=(100, 150, 80)
)

variants = create_image_variants(sample)

print("Number of variants:", len(variants))

for name, image in variants.items():

    print(
        name,
        "->",
        image.size,
        image.mode
    )

assert len(variants) == 6

print("\nALL MODULE TESTS PASSED!")
