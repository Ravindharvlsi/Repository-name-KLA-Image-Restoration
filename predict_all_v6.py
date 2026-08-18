import os
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, r"C:\KLA\code")

from model_v5 import V5RestorationNet


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = r"C:\KLA\train\train\NoisyLR"

CHECKPOINT = (
    r"C:\KLA\results\model_v6"
    r"\checkpoints\best_model_v6.pth"
)

OUTPUT_DIR = (
    r"C:\KLA\results\model_v6"
    r"\outputs\all_restored"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("V6 - ALL IMAGE PREDICTION")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MODEL
# ============================================================

model = V5RestorationNet(
    in_channels=1,
    out_channels=1,
    features=80,
    num_blocks=24
).to(device)


print("Loading V6 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("V6 model loaded.")
print(
    "Best PSNR:",
    checkpoint.get("psnr", "N/A")
)
print(
    "Best SSIM:",
    checkpoint.get("ssim", "N/A")
)


# ============================================================
# FILES
# ============================================================

files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".npy")
])

print()
print("Total images:", len(files))
print("Output:", OUTPUT_DIR)
print()


# ============================================================
# PROCESS ALL
# ============================================================

with torch.no_grad():

    for i, filename in enumerate(files, 1):

        input_path = os.path.join(
            INPUT_DIR,
            filename
        )

        # Load NoisyLR
        noisy = np.load(
            input_path
        ).astype(np.float32)

        # Tensor: 1 x 1 x H x W
        tensor = torch.from_numpy(
            noisy
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(device)

        # Prediction
        output = model(tensor)

        output = (
            output
            .squeeze()
            .cpu()
            .numpy()
        )

        output = np.clip(
            output,
            0.0,
            1.0
        )

        # Convert to 8-bit image
        image_uint8 = (
            output * 255.0
        ).round().astype(
            np.uint8
        )

        # Output filename
        name = os.path.splitext(
            filename
        )[0]

        output_path = os.path.join(
            OUTPUT_DIR,
            name + "_V6.png"
        )

        Image.fromarray(
            image_uint8
        ).save(
            output_path
        )

        # Progress
        if i % 50 == 0 or i == 1:

            print(
                f"[{i}/{len(files)}] "
                f"Saved: {name}_V6.png"
            )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("ALL V6 PREDICTIONS COMPLETE")
print("=" * 70)

print(
    "Images processed:",
    len(files)
)

print(
    "Saved to:"
)

print(
    OUTPUT_DIR
)

print("=" * 70)