
import math
import torch
import torch.nn.functional as F


def _validate_inputs(logits, labels):
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")

    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")

    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")

    if labels.ndim != 1:
        raise ValueError("labels must have shape (N,)")

    if logits.shape[0] != labels.shape[0]:
        raise ValueError("Logits and labels must have matching samples")

    if logits.shape[0] == 0:
        raise ValueError("Input cannot be empty")

    if logits.shape[1] < 2:
        raise ValueError("At least two classes are required")

    if not torch.isfinite(logits).all():
        raise ValueError("Logits contain NaN or infinite values")

    labels = labels.to(device=logits.device, dtype=torch.long)

    if torch.any(labels < 0) or torch.any(labels >= logits.shape[1]):
        raise ValueError("Labels contain invalid class indices")

    return labels


def calculate_calibration_metrics(logits, labels, n_bins=15):
    """
    Calculate multiclass calibration metrics.

    Returns:
        ECE: Expected Calibration Error
        MCE: Maximum Calibration Error
        Brier Score: Multiclass probability error
        NLL: Negative Log-Likelihood
        bins: Confidence-bin statistics

    Metrics are computed on CPU in float64 for numerical
    consistency and compatibility with Apple MPS.
    """

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    labels = _validate_inputs(logits, labels)

    # Move evaluation tensors to CPU.
    # This avoids unsupported float64 operations on MPS.
    evaluation_logits = logits.detach().to(
        device="cpu",
        dtype=torch.float64
    )

    evaluation_labels = labels.detach().to(
        device="cpu",
        dtype=torch.long
    )

    probabilities = torch.softmax(evaluation_logits, dim=1)

    confidences, predictions = probabilities.max(dim=1)

    accuracies = predictions.eq(evaluation_labels).to(torch.float64)

    # Expected Calibration Error and Maximum Calibration Error
    boundaries = torch.linspace(
        0.0,
        1.0,
        n_bins + 1,
        dtype=torch.float64
    )

    ece = 0.0
    mce = 0.0
    bin_results = []

    total_samples = len(evaluation_labels)

    for i in range(n_bins):

        lower = boundaries[i]
        upper = boundaries[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower) &
                (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower) &
                (confidences <= upper)
            )

        count = int(mask.sum().item())

        if count == 0:
            continue

        avg_confidence = confidences[mask].mean()
        avg_accuracy = accuracies[mask].mean()

        gap = abs(
            avg_confidence.item() -
            avg_accuracy.item()
        )

        ece += (count / total_samples) * gap
        mce = max(mce, gap)

        bin_results.append({
            "bin": i + 1,
            "count": count,
            "confidence": round(avg_confidence.item(), 6),
            "accuracy": round(avg_accuracy.item(), 6),
            "gap": round(gap, 6)
        })

    # Multiclass Brier Score
    one_hot_labels = F.one_hot(
        evaluation_labels,
        num_classes=evaluation_logits.shape[1]
    ).to(torch.float64)

    brier_score = torch.sum(
        (probabilities - one_hot_labels) ** 2,
        dim=1
    ).mean()

    # Negative Log-Likelihood
    nll = F.cross_entropy(
        evaluation_logits,
        evaluation_labels
    )

    accuracy = accuracies.mean()

    return {
        "sample_count": total_samples,
        "accuracy": round(accuracy.item(), 6),
        "ece": round(ece, 6),
        "mce": round(mce, 6),
        "brier_score": round(brier_score.item(), 6),
        "nll": round(nll.item(), 6),
        "bins": bin_results
    }


def fit_temperature(validation_logits, validation_labels):
    """
    Fit temperature using validation data only.

    Uses bounded sigmoid parameterization and CPU float64
    optimization to avoid MPS float64 limitations.

    Temperature range: 0.05 to 10.0
    """

    labels = _validate_inputs(
        validation_logits,
        validation_labels
    )

    logits = validation_logits.detach().to(
        device="cpu",
        dtype=torch.float64
    )

    labels = labels.detach().to(
        device="cpu",
        dtype=torch.long
    )

    minimum_temperature = 0.05
    maximum_temperature = 10.0

    # Initialize temperature at 1.0.
    initial_ratio = (
        (1.0 - minimum_temperature) /
        (maximum_temperature - minimum_temperature)
    )

    initial_parameter = math.log(
        initial_ratio / (1.0 - initial_ratio)
    )

    parameter = torch.nn.Parameter(
        torch.tensor(
            initial_parameter,
            dtype=torch.float64
        )
    )

    optimizer = torch.optim.LBFGS(
        [parameter],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe"
    )

    def get_temperature():
        return minimum_temperature + (
            maximum_temperature - minimum_temperature
        ) * torch.sigmoid(parameter)

    def closure():
        optimizer.zero_grad()

        temperature = get_temperature()

        loss = F.cross_entropy(
            logits / temperature,
            labels
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    temperature = get_temperature().detach().item()

    return float(temperature)


def calibrate_logits(logits, temperature):
    if not math.isfinite(float(temperature)):
        raise ValueError("Temperature must be finite")

    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    return logits / float(temperature)


def get_prediction_confidence(logits, temperature=1.0):
    calibrated_logits = calibrate_logits(
        logits,
        temperature
    )

    probabilities = torch.softmax(
        calibrated_logits,
        dim=1
    )

    confidence, predicted_class = probabilities.max(dim=1)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }
