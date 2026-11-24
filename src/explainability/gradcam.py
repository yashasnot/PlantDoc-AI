
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def generate_gradcam(model, image_tensor, target_class=None):

    model.eval()

    target_layer = model.features[-1]

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):

        activations["value"] = output

        output.register_hook(
            lambda grad: gradients.update(
                {"value": grad}
            )
        )

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        model.zero_grad(set_to_none=True)

        with torch.enable_grad():

            output = model(image_tensor)

            probabilities = torch.softmax(
                output, dim=1
            )

            predicted_class = output.argmax(1).item()

            if target_class is None:
                target_class = predicted_class

            output[0, target_class].backward()

        features = activations["value"].detach()
        grads = gradients["value"].detach()

        weights = grads.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = F.relu(
            (weights * features).sum(dim=1)
        )[0].cpu().numpy()

        cam = cv2.resize(
            cam,
            (224, 224)
        )

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return (
            cam,
            predicted_class,
            probabilities[0, predicted_class].item()
        )

    finally:
        handle.remove()
