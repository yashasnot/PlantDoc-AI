
import torch
import torch.nn.functional as F

from PIL import Image, ImageEnhance, ImageFilter


# ==================================================
# PLANTDOC AI - RELIABILITY GATE V2
# MPS / CUDA / CPU compatible
# ==================================================

DEFAULT_TEMPERATURE = 0.809876024723053

CONFIDENCE_THRESHOLD = 0.95
STABILITY_THRESHOLD = 1.0


def create_image_variants(image):
    """Generate six controlled image variants."""

    image = image.convert("RGB")

    return {
        "original": image,

        "brightness_low": ImageEnhance.Brightness(
            image
        ).enhance(0.85),

        "brightness_high": ImageEnhance.Brightness(
            image
        ).enhance(1.15),

        "contrast_low": ImageEnhance.Contrast(
            image
        ).enhance(0.85),

        "contrast_high": ImageEnhance.Contrast(
            image
        ).enhance(1.15),

        "mild_blur": image.filter(
            ImageFilter.GaussianBlur(radius=0.7)
        )
    }


def assess_prediction_reliability(
    model,
    image,
    transform,
    device,
    temperature=DEFAULT_TEMPERATURE
):
    """
    Evaluate disease prediction reliability.

    Confidence:
        Softmax(logits / temperature)

    Stability:
        Agreement across six controlled variants.

    Uses float32 for compatibility with Apple MPS.
    """

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    variants = create_image_variants(image)
    names = list(variants.keys())

    # Transform all variants
    tensors = torch.stack([
        transform(variants[name])
        for name in names
    ])

    # Ensure model input uses float32
    tensors = tensors.to(
        device=device,
        dtype=torch.float32
    )

    model.eval()

    with torch.inference_mode():

        logits = model(tensors)

        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "Model must return a torch.Tensor."
            )

        if logits.ndim != 2:
            raise ValueError(
                "Expected logits with shape [batch, classes]."
            )

        if logits.shape[0] != len(names):
            raise ValueError(
                "Model output batch size does not match variants."
            )

        # Keep probability calculations MPS-compatible
        logits = logits.float()

        raw_probs = F.softmax(
            logits,
            dim=1
        )

        calibrated_logits = (
            logits / float(temperature)
        )

        calibrated_probs = F.softmax(
            calibrated_logits,
            dim=1
        )

    # Original image is index zero
    original_raw = raw_probs[0]
    original_calibrated = calibrated_probs[0]

    raw_confidence, raw_prediction = (
        original_raw.max(dim=0)
    )

    calibrated_confidence, prediction = (
        original_calibrated.max(dim=0)
    )

    # Predictions across all variants
    all_predictions = calibrated_probs.argmax(
        dim=1
    )

    original_prediction = int(
        prediction.item()
    )

    agreement_count = int(
        (
            all_predictions == prediction
        ).sum().item()
    )

    stability = (
        agreement_count / len(names)
    )

    # Top five predictions
    top_probs, top_indices = (
        original_calibrated.topk(
            min(5, original_calibrated.numel())
        )
    )

    # Top-two probability margin
    sorted_probs = original_calibrated.sort(
        descending=True
    ).values

    if sorted_probs.numel() > 1:
        margin = float(
            (
                sorted_probs[0] -
                sorted_probs[1]
            ).item()
        )
    else:
        margin = 1.0

    # Reliability conditions
    confidence_pass = (
        float(calibrated_confidence.item())
        >= CONFIDENCE_THRESHOLD
    )

    stability_pass = (
        stability >= STABILITY_THRESHOLD
    )

    if confidence_pass and stability_pass:

        decision = "Supported"

        explanation = (
            "The prediction meets the calibrated "
            "confidence threshold and remains consistent "
            "across all six controlled image variants."
        )

    elif confidence_pass or stability_pass:

        decision = "Provisional"

        explanation = (
            "Only one reliability condition is satisfied. "
            "Human verification is recommended."
        )

    else:

        decision = "Abstain"

        explanation = (
            "Neither reliability condition is satisfied. "
            "The system recommends additional verification."
        )

    return {

        "decision": decision,

        "prediction_index": original_prediction,

        "raw_prediction_index": int(
            raw_prediction.item()
        ),

        "raw_confidence": float(
            raw_confidence.item()
        ),

        "calibrated_confidence": float(
            calibrated_confidence.item()
        ),

        "stability": float(stability),

        "agreement_count": agreement_count,

        "total_variants": len(names),

        "probability_margin": margin,

        "top5_indices": (
            top_indices.detach().cpu().tolist()
        ),

        "top5_probabilities": (
            top_probs.detach().cpu().tolist()
        ),

        "variant_predictions": {
            name: int(
                all_predictions[i].item()
            )
            for i, name in enumerate(names)
        },

        "variant_confidences": {
            name: float(
                calibrated_probs[i].max().item()
            )
            for i, name in enumerate(names)
        },

        "confidence_pass": confidence_pass,

        "stability_pass": stability_pass,

        "temperature": float(temperature),

        "thresholds": {
            "confidence": CONFIDENCE_THRESHOLD,
            "stability": STABILITY_THRESHOLD
        },

        "explanation": explanation
    }
