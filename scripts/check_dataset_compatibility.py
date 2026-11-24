
import os
import json
import torch

DATASET_DIR = os.path.expanduser(
    "~/Downloads/PlantVillage/PlantVillage"
)

CHECKPOINT_PATH = (
    "models/classification/efficientnet_b0_best.pth"
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def normalize_class_name(name):
    """
    Normalize naming differences for comparison.
    This does not change the actual class labels.
    """
    return "".join(
        character.lower()
        for character in name
        if character.isalnum()
    )


def main():

    print("\n" + "=" * 65)
    print("PLANTDOC AI - DATASET COMPATIBILITY AUDIT")
    print("=" * 65)

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model_classes = checkpoint.get("class_names", [])

    if not model_classes:
        raise ValueError(
            "Checkpoint does not contain class_names."
        )

    dataset_classes = sorted([
        name
        for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    ])

    dataset_counts = {}

    for class_name in dataset_classes:

        class_path = os.path.join(
            DATASET_DIR,
            class_name
        )

        count = 0

        for root, _, files in os.walk(class_path):
            count += sum(
                file.lower().endswith(IMAGE_EXTENSIONS)
                for file in files
            )

        dataset_counts[class_name] = count

    normalized_model = {
        normalize_class_name(name): name
        for name in model_classes
    }

    normalized_dataset = {
        normalize_class_name(name): name
        for name in dataset_classes
    }

    matched = sorted(
        set(normalized_model) & set(normalized_dataset)
    )

    missing_from_dataset = sorted(
        set(normalized_model) - set(normalized_dataset)
    )

    extra_in_dataset = sorted(
        set(normalized_dataset) - set(normalized_model)
    )

    print("\nMODEL INFORMATION")
    print("Number of model classes:", len(model_classes))
    print("Checkpoint validation accuracy:",
          checkpoint.get("validation_accuracy", "Not available"))

    print("\nDATASET INFORMATION")
    print("Number of dataset classes:", len(dataset_classes))
    print("Total images:", sum(dataset_counts.values()))

    print("\nDATASET CLASS COUNTS")

    for name, count in sorted(dataset_counts.items()):
        print(f"{name}: {count}")

    print("\nCOMPATIBILITY RESULTS")

    print("Matched classes:", len(matched))
    print("Missing model classes:", len(missing_from_dataset))
    print("Extra dataset classes:", len(extra_in_dataset))

    print("\nMATCHED CLASSES")

    for key in matched:
        print(
            f"Model: {normalized_model[key]} "
            f"<-> Dataset: {normalized_dataset[key]}"
        )

    print("\nMODEL CLASSES MISSING FROM DATASET")

    for key in missing_from_dataset:
        print(normalized_model[key])

    print("\nEXTRA DATASET CLASSES")

    for key in extra_in_dataset:
        print(normalized_dataset[key])

    report = {
        "model_class_count": len(model_classes),
        "dataset_class_count": len(dataset_classes),
        "total_dataset_images": sum(dataset_counts.values()),
        "matched_class_count": len(matched),
        "missing_model_classes": [
            normalized_model[key]
            for key in missing_from_dataset
        ],
        "extra_dataset_classes": [
            normalized_dataset[key]
            for key in extra_in_dataset
        ],
        "dataset_class_counts": dataset_counts
    }

    os.makedirs("reports", exist_ok=True)

    with open(
        "reports/dataset_compatibility.json",
        "w"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nReport saved to:")
    print("reports/dataset_compatibility.json")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
