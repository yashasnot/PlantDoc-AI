import json
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image

from evaluate_classifier import (
    ROOT,
    CHECKPOINT_PATH,
    SPLIT_CSV,
    DEVICE,
    BATCH_SIZE,
    PlantVillageDataset,
    transform
)

from src.reliability.gate import (
    assess_prediction_reliability,
    DEFAULT_TEMPERATURE
)


def main():
    print("\nPLANTDOC AI - FULL RELIABILITY EVALUATION")

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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    dataset = PlantVillageDataset(SPLIT_CSV)

    records = []

    for index, sample in enumerate(dataset.samples):
        image_path = sample["image_path"]
        true_index = int(sample["class_index"])

        image = Image.open(image_path).convert("RGB")

        result = assess_prediction_reliability(
            model=model,
            image=image,
            transform=transform,
            device=DEVICE,
            temperature=DEFAULT_TEMPERATURE
        )

        predicted_index = result["prediction_index"]

        records.append({
            "image_path": image_path,
            "true_class": classes[true_index],
            "predicted_class": classes[predicted_index],
            "is_correct": true_index == predicted_index,
            "decision": result["decision"],
            "raw_confidence": result["raw_confidence"],
            "calibrated_confidence": result["calibrated_confidence"],
            "stability": result["stability"],
            "agreement_count": result["agreement_count"],
            "confidence_pass": result["confidence_pass"],
            "stability_pass": result["stability_pass"]
        })

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(dataset)} images")

    df = pd.DataFrame(records)

    output = ROOT / "reports/subset_reliability_predictions.csv"
    df.to_csv(output, index=False)

    print("\nRELIABILITY RESULTS")

    for decision in ["Supported", "Provisional", "Abstain"]:
        group = df[df["decision"] == decision]

        print(f"\n{decision}")
        print("Samples:", len(group))
        print("Coverage:", round(len(group) / len(df) * 100, 2), "%")

        if len(group):
            print(
                "Accuracy:",
                round(group["is_correct"].mean() * 100, 2),
                "%"
            )

    accepted = df[df["decision"].isin(["Supported", "Provisional"])]

    print("\nOVERALL")
    print("Total samples:", len(df))
    print("Overall accuracy:", round(df["is_correct"].mean() * 100, 2), "%")
    print("Accepted samples:", len(accepted))

    if len(accepted):
        print(
            "Accepted accuracy:",
            round(accepted["is_correct"].mean() * 100, 2),
            "%"
        )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
