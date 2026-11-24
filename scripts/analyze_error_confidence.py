import json
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import models

from evaluate_classifier import (
    ROOT,
    CHECKPOINT_PATH,
    SPLIT_CSV,
    DEVICE,
    BATCH_SIZE,
    PlantVillageDataset,
    transform
)

def main():
    print("\nPLANTDOC AI - ERROR CONFIDENCE ANALYSIS")

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

    dataset = PlantVillageDataset(SPLIT_CSV, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    records = []

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            logits = model(images.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1).cpu()

            top2_probs, top2_indices = torch.topk(
                probabilities, k=2, dim=1
            )

            for i in range(len(labels)):
                true_idx = int(labels[i])
                pred_idx = int(top2_indices[i, 0])

                records.append({
                    "true_class": classes[true_idx],
                    "predicted_class": classes[pred_idx],
                    "is_correct": true_idx == pred_idx,
                    "prediction_confidence": float(top2_probs[i, 0]),
                    "actual_class_probability": float(
                        probabilities[i, true_idx]
                    ),
                    "second_class": classes[int(top2_indices[i, 1])],
                    "second_class_probability": float(top2_probs[i, 1]),
                    "confidence_margin": float(
                        top2_probs[i, 0] - top2_probs[i, 1]
                    )
                })

            if (batch_idx + 1) % 20 == 0:
                print(f"Processed {batch_idx + 1}/{len(loader)} batches")

    df = pd.DataFrame(records)

    # Save confidence predictions for the complete validation set.
    all_output = ROOT / "reports/subset_all_confidence_predictions.csv"
    df.to_csv(all_output, index=False)

    # Compare confidence for correct and incorrect predictions.
    print("\\nALL VALIDATION CONFIDENCE SUMMARY")

    print(
        df.groupby("is_correct")["prediction_confidence"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(4)
    )

    # Evaluate selective prediction at different thresholds.
    threshold_results = []
    total_errors = (~df["is_correct"]).sum()

    for threshold in [0.50, 0.70, 0.90, 0.95]:
        accepted = df["prediction_confidence"] >= threshold
        accepted_df = df[accepted]

        threshold_results.append({
            "threshold": threshold,
            "accepted_samples": len(accepted_df),
            "coverage": len(accepted_df) / len(df),
            "accepted_accuracy": (
                accepted_df["is_correct"].mean()
                if len(accepted_df) else 0
            ),
            "rejected_errors": (
                ((~df["is_correct"]) & (~accepted)).sum()
            ),
            "error_rejection_rate": (
                ((~df["is_correct"]) & (~accepted)).sum()
                / total_errors if total_errors else 0
            )
        })

    threshold_df = pd.DataFrame(threshold_results)

    threshold_output = (
        ROOT / "reports/subset_confidence_threshold_analysis.csv"
    )
    threshold_df.to_csv(threshold_output, index=False)

    print("\\nCONFIDENCE THRESHOLD ANALYSIS")
    print(threshold_df.round(4).to_string(index=False))

    print("\\nSaved:", all_output)
    print("Saved:", threshold_output)

    pairs = [
        ("Tomato___Late_blight", "Tomato___Early_blight"),
        (
            "Tomato___Target_Spot",
            "Tomato___Spider_mites Two-spotted_spider_mite"
        )
    ]

    selected = []

    for actual, predicted in pairs:
        errors = df[
            (df["true_class"] == actual) &
            (df["predicted_class"] == predicted)
        ]
        selected.append(errors)

    result = pd.concat(selected, ignore_index=True)

    for column in [
        "prediction_confidence",
        "actual_class_probability",
        "second_class_probability",
        "confidence_margin"
    ]:
        result[column] = (result[column] * 100).round(3)

    output = ROOT / "reports/subset_error_confidence_analysis.csv"
    result.to_csv(output, index=False)

    print("\nSELECTED ERRORS:", len(result))
    print("\nCONFIDENCE SUMMARY")

    print(
        result.groupby(["true_class", "predicted_class"])[
            ["prediction_confidence",
             "actual_class_probability",
             "confidence_margin"]
        ].mean().round(3)
    )

    print("\nINDIVIDUAL ERROR ANALYSIS")
    print(result.to_string(index=False))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
