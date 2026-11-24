
import os
import sys

# Add project root to Python path
ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from src.reliability.confidence import (
    calculate_calibration_metrics,
    fit_temperature,
    calibrate_logits,
    get_prediction_confidence
)


def main():

    print("\n" + "=" * 55)
    print("PLANTDOC AI - CONFIDENCE CALIBRATION TEST")
    print("=" * 55)

    # Synthetic test data for checking module functionality.
    # These are NOT actual model evaluation results.
    torch.manual_seed(42)

    logits = torch.randn(100, 5) * 2

    labels = torch.randint(
        low=0,
        high=5,
        size=(100,)
    )

    print("\n1. Testing original calibration metrics...")

    original_metrics = calculate_calibration_metrics(
        logits,
        labels,
        n_bins=15
    )

    print("Original ECE:", original_metrics["ece"])
    print("Original MCE:", original_metrics["mce"])
    print("Occupied bins:", len(original_metrics["bins"]))

    print("\n2. Fitting temperature on validation-like data...")

    temperature = fit_temperature(
        logits,
        labels
    )

    print("Learned temperature:", round(temperature, 4))

    print("\n3. Applying temperature scaling...")

    calibrated_logits = calibrate_logits(
        logits,
        temperature
    )

    calibrated_metrics = calculate_calibration_metrics(
        calibrated_logits,
        labels,
        n_bins=15
    )

    print("Calibrated ECE:", calibrated_metrics["ece"])
    print("Calibrated MCE:", calibrated_metrics["mce"])

    print("\n4. Checking prediction confidence...")

    prediction = get_prediction_confidence(
        logits,
        temperature
    )

    print(
        "First predicted class:",
        prediction["predicted_class"][0].item()
    )

    print(
        "First prediction confidence:",
        round(prediction["confidence"][0].item(), 4)
    )

    print("\n5. Checking class prediction consistency...")

    original_predictions = torch.argmax(logits, dim=1)

    calibrated_predictions = torch.argmax(
        calibrated_logits,
        dim=1
    )

    predictions_unchanged = torch.equal(
        original_predictions,
        calibrated_predictions
    )

    print("Predictions unchanged:", predictions_unchanged)

    print("\n" + "=" * 55)
    print("TEST COMPLETED")
    print("=" * 55)

    print(
        "\nNOTE: Synthetic data verifies code functionality only. "
        "Real calibration results must be calculated using "
        "the trained model and held-out validation data."
    )


if __name__ == "__main__":
    main()
