
import os
import json
import torch
import torch.nn as nn
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    ROOT, "models", "classification", "final_classifier.pth"
)

LABEL_PATH = os.path.join(
    ROOT, "models", "classification", "class_names.json"
)

print("=" * 55)
print("PLANTDOC AI - CLASSIFIER VERIFICATION")
print("=" * 55)

# 1. Select available device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Selected device:", device)

# 2. Load class labels
with open(LABEL_PATH, "r") as f:
    class_names = json.load(f)

print("Number of classes:", len(class_names))

# 3. Reconstruct EfficientNet-B0
model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(
    in_features, len(class_names)
)

# 4. Load trained weights
state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Checkpoint loaded successfully!")

# 5. Test inference with a dummy image
dummy_image = torch.randn(
    1, 3, 224, 224
).to(device)

with torch.no_grad():
    output = model(dummy_image)
    probabilities = torch.softmax(output, dim=1)

predicted_index = probabilities.argmax(dim=1).item()
confidence = probabilities[0, predicted_index].item()

print("\nINFERENCE TEST")
print("Output shape:", tuple(output.shape))
print("Predicted class:", class_names[predicted_index])
print("Confidence:", round(confidence * 100, 2), "%")

assert output.shape[1] == len(class_names)

print("\nSUCCESS: CLASSIFIER VERIFICATION COMPLETE")
