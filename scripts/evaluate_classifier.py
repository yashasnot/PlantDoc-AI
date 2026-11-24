
import os
import sys
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reliability.confidence import (
    calculate_calibration_metrics,
    fit_temperature,
    calibrate_logits
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = (
    ROOT / "models/classification/efficientnet_b0_best.pth"
)

SPLIT_CSV = ROOT / "reports/plantvillage_split.csv"

OUTPUT_PATH = ROOT / "reports/subset_15class_evaluation.json"

BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_SIZE = 224


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class PlantVillageDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.samples = []
        self.transform = transform

        with open(csv_path, "r", encoding="utf-8") as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["split"] != "validation":
                    continue

                self.samples.append({
                    "image_path": row["image_path"],
                    "class_index": int(row["class_index"]),
                    "class_name": row["class_name"]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = sample["class_index"]

        return image, label


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model():

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    classes = checkpoint["class_names"]

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("\nMODEL LOADED SUCCESSFULLY")
    print("Architecture: EfficientNet-B0")
    print("Number of classes:", len(classes))
    print("Device:", DEVICE)
    print(
        "Checkpoint validation accuracy:",
        checkpoint.get("validation_accuracy")
    )

    return model, classes


# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# VALIDATION INFERENCE
# --------------------------------------------------

def generate_validation_logits(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    all_logits = []
    all_labels = []

    print("\nGENERATING VALIDATION PREDICTIONS")

    model.eval()

    with torch.inference_mode():

        for batch_index, (images, labels) in enumerate(loader):

            images = images.to(DEVICE)

            logits = model(images)

            all_logits.append(
                logits.detach().cpu()
            )

            all_labels.append(
                labels.detach().cpu()
            )

            if (batch_index + 1) % 20 == 0:
                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/{len(loader)}"
                )

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return logits, labels


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_nll(logits, labels):

    return F.cross_entropy(
        logits,
        labels
    ).item()


def calculate_accuracy(logits, labels):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    return (
        predictions.eq(labels).float().mean().item()
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("PLANTDOC AI - EFFICIENTNET-B0 VALIDATION EVALUATION")
    print("=" * 65)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(SPLIT_CSV)

    model, classes = load_model()

    dataset = PlantVillageDataset(
        SPLIT_CSV,
        transform=transform
    )

    if len(dataset) == 0:
        raise ValueError("No validation images found.")

    print("\nValidation images:", len(dataset))

    # Generate actual model predictions.
    logits, labels = generate_validation_logits(
        model,
        dataset
    )

    # ----------------------------------------------
    # BEFORE CALIBRATION
    # ----------------------------------------------

    original_accuracy = calculate_accuracy(
        logits,
        labels
    )

    original_nll = calculate_nll(
        logits,
        labels
    )

    original_calibration = calculate_calibration_metrics(
        logits,
        labels,
        n_bins=15
    )

    # ----------------------------------------------
    # FIT TEMPERATURE USING VALIDATION DATA
    # ----------------------------------------------

    print("\nUSING EXISTING FULL-MODEL TEMPERATURE...")

    # Preserve the previously established calibration temperature.
    temperature = 0.809876

    calibrated_logits = calibrate_logits(
        logits,
        temperature
    )

    # ----------------------------------------------
    # AFTER CALIBRATION
    # ----------------------------------------------

    calibrated_accuracy = calculate_accuracy(
        calibrated_logits,
        labels
    )

    calibrated_nll = calculate_nll(
        calibrated_logits,
        labels
    )

    calibrated_calibration = calculate_calibration_metrics(
        calibrated_logits,
        labels,
        n_bins=15
    )

    # ----------------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)

    print("\nBEFORE TEMPERATURE SCALING")

    print(f"Accuracy: {original_accuracy * 100:.4f}%")
    print(f"NLL: {original_nll:.6f}")
    print(f"ECE: {original_calibration['ece']:.6f}")
    print(f"MCE: {original_calibration['mce']:.6f}")

    print("\nTEMPERATURE SCALING")

    print(f"Learned Temperature: {temperature:.6f}")

    print("\nAFTER TEMPERATURE SCALING")

    print(f"Accuracy: {calibrated_accuracy * 100:.4f}%")
    print(f"NLL: {calibrated_nll:.6f}")
    print(f"ECE: {calibrated_calibration['ece']:.6f}")
    print(f"MCE: {calibrated_calibration['mce']:.6f}")

    print("\nCALIBRATION IMPROVEMENT")

    print(
        "ECE change:",
        round(
            original_calibration["ece"]
            - calibrated_calibration["ece"],
            6
        )
    )

    print(
        "NLL change:",
        round(
            original_nll - calibrated_nll,
            6
        )
    )

    # Verify temperature scaling does not change argmax.
    predictions_unchanged = torch.equal(
        torch.argmax(logits, dim=1),
        torch.argmax(calibrated_logits, dim=1)
    )

    print(
        "\nPredicted classes unchanged:",
        predictions_unchanged
    )

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    results = {
        "model": "EfficientNet-B0",
        "checkpoint": "efficientnet_b0_best.pth",
        "evaluation_split": "validation",
        "evaluation_scope": "15 represented classes; 38-output classifier",
        "number_of_model_classes": len(classes),
        "number_of_evaluated_samples": len(dataset),
        "temperature": temperature,
        "before_calibration": {
            "accuracy": original_accuracy,
            "nll": original_nll,
            "ece": original_calibration["ece"],
            "mce": original_calibration["mce"],
            "bins": original_calibration["bins"]
        },
        "after_calibration": {
            "accuracy": calibrated_accuracy,
            "nll": calibrated_nll,
            "ece": calibrated_calibration["ece"],
            "mce": calibrated_calibration["mce"],
            "bins": calibrated_calibration["bins"]
        },
        "predictions_unchanged": predictions_unchanged,
        "class_names": classes
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nRESULTS SAVED TO:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 65)
    print("VALIDATION EVALUATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
