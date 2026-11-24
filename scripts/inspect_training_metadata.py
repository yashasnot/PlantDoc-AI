
import torch

path = "models/classification/efficientnet_b0_best.pth"

checkpoint = torch.load(
    path,
    map_location="cpu",
    weights_only=True
)

print("\n===== MODEL TRAINING METADATA =====")

for key in [
    "num_classes",
    "image_size",
    "validation_accuracy"
]:
    print(f"{key}: {checkpoint.get(key, 'Not available')}")

print("\n===== CHECKPOINT CLASS NAMES =====")

saved_classes = checkpoint.get("class_names", [])

print("Number of classes:", len(saved_classes))

for index, name in enumerate(saved_classes):
    print(index, name)

