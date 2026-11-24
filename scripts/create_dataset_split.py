
import os
import csv
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import torch


# -----------------------------
# Configuration
# -----------------------------

DATASET_DIR = Path.home() / "Downloads/PlantVillage/PlantVillage"

CHECKPOINT_PATH = Path(
    "models/classification/efficientnet_b0_best.pth"
)

OUTPUT_CSV = Path("reports/plantvillage_split.csv")
OUTPUT_JSON = Path("reports/plantvillage_split_summary.json")

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def normalize_class_name(name):
    return "".join(
        character.lower()
        for character in name
        if character.isalnum()
    )


def main():

    random.seed(SEED)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_DIR}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model_classes = checkpoint["class_names"]

    # Map normalized class names to original checkpoint indices.
    class_mapping = {
        normalize_class_name(name): index
        for index, name in enumerate(model_classes)
    }

    grouped_images = defaultdict(list)

    # Read the class-wise dataset.
    for class_folder in sorted(DATASET_DIR.iterdir()):

        if not class_folder.is_dir():
            continue

        normalized_name = normalize_class_name(
            class_folder.name
        )

        if normalized_name not in class_mapping:
            print(
                "Skipping unmatched class:",
                class_folder.name
            )
            continue

        class_index = class_mapping[normalized_name]

        for image_path in class_folder.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                grouped_images[class_folder.name].append(
                    (str(image_path.resolve()), class_index)
                )

    rows = []
    summary = {
        "random_seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "model_total_classes": len(model_classes),
        "evaluated_classes": {},
        "split_counts": {
            "train": 0,
            "validation": 0,
            "test": 0
        }
    }

    # Create a deterministic split independently for each class.
    for class_name, images in sorted(grouped_images.items()):

        random.shuffle(images)

        total = len(images)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        split_groups = {
            "train": train_images,
            "validation": val_images,
            "test": test_images
        }

        summary["evaluated_classes"][class_name] = {
            "class_index": images[0][1],
            "total": total,
            "train": len(train_images),
            "validation": len(val_images),
            "test": len(test_images)
        }

        for split_name, split_images in split_groups.items():

            for image_path, class_index in split_images:

                rows.append({
                    "image_path": image_path,
                    "class_name": class_name,
                    "class_index": class_index,
                    "split": split_name
                })

                summary["split_counts"][split_name] += 1

    if not rows:
        raise RuntimeError("No compatible images found.")

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "class_name",
                "class_index",
                "split"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    summary["total_images"] = len(rows)

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 60)
    print("PLANTDOC AI - DATASET SPLIT CREATED")
    print("=" * 60)

    print("Random seed:", SEED)
    print("Evaluated classes:", len(grouped_images))
    print("Total images:", len(rows))

    print("\nSPLIT COUNTS")

    for split, count in summary["split_counts"].items():
        print(f"{split}: {count}")

    print("\nCLASS-WISE DISTRIBUTION")

    for name, details in summary["evaluated_classes"].items():
        print(
            f"{name}: "
            f"Train={details['train']}, "
            f"Val={details['validation']}, "
            f"Test={details['test']}"
        )

    print("\nSaved files:")
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
