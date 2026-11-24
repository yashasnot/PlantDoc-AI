import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from torch.utils.data import DataLoader
from torchvision import models

from evaluate_classifier import (
    ROOT,
    CHECKPOINT_PATH,
    SPLIT_CSV,
    DEVICE,
    BATCH_SIZE,
    IMAGE_SIZE,
    PlantVillageDataset,
    transform
)

OUTPUT_DIR = ROOT / "reports"

def main():
    print("\nPLANTDOC AI - SUBSET PERFORMANCE ANALYSIS")

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

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for i, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions)

            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(loader)} batches")

    # Identify the 15 classes actually represented in this split.
    represented_indices = sorted(set(y_true))

    # Keep all 38 model output classes visible in the confusion matrix.
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes)))
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=represented_indices,
        target_names=[classes[i] for i in represented_indices],
        output_dict=True,
        zero_division=0
    )

    summary = {
        "evaluation_scope": "15 represented classes; 38-output classifier",
        "samples": len(y_true),
        "model_output_classes": len(classes),
        "represented_classes": len(represented_indices),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=represented_indices,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred,
            labels=represented_indices,
            average="weighted",
            zero_division=0
        ),
        "per_class": report
    }

    with open(OUTPUT_DIR / "subset_15class_classification_report.json", "w") as f:
        json.dump(summary, f, indent=4)

    pd.DataFrame(
        matrix,
        index=classes,
        columns=classes
    ).to_csv(OUTPUT_DIR / "subset_15class_confusion_matrix.csv")

    pd.DataFrame({
        "true_class_index": y_true,
        "true_class": [classes[i] for i in y_true],
        "predicted_class_index": y_pred,
        "predicted_class": [classes[i] for i in y_pred]
    }).to_csv(
        OUTPUT_DIR / "subset_15class_predictions.csv",
        index=False
    )

    print("\nAccuracy:", round(summary["accuracy"] * 100, 4), "%")
    print("Macro F1:", round(summary["macro_f1"], 4))
    print("Weighted F1:", round(summary["weighted_f1"], 4))
    print("\nPER-CLASS RESULTS")

    for name, values in report.items():
        if isinstance(values, dict) and "f1-score" in values:
            print(
                f"{name}: "
                f"Precision={values['precision']:.4f}, "
                f"Recall={values['recall']:.4f}, "
                f"F1={values['f1-score']:.4f}, "
                f"Support={values['support']}"
            )

    print("\nSaved:")
    print("reports/subset_15class_classification_report.json")
    print("reports/subset_15class_confusion_matrix.csv")
    print("reports/subset_15class_predictions.csv")


if __name__ == "__main__":
    main()
