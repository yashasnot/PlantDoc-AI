
import torch
import os

checkpoint_paths = [
    "models/classification/efficientnet_b0_best.pth",
    "models/classification/final_classifier.pth"
]

for path in checkpoint_paths:
    print("\n" + "=" * 60)
    print("FILE:", path)

    if not os.path.exists(path):
        print("File not found")
        continue

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    print("Type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("Top-level keys:", list(checkpoint.keys())[:30])

        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint

        if isinstance(state, dict):
            print("Number of parameters:", len(state))
            print("First 15 parameter names:")

            for key in list(state.keys())[:15]:
                print(key)
