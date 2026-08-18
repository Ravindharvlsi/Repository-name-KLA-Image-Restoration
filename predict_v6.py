import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\KLA\code")

from model_v5 import V5RestorationNet


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = r"C:\KLA\train\train\NoisyLR"
GT_DIR = r"C:\KLA\train\train\GT"

CHECKPOINT = (
    r"C:\KLA\results\model_v6"
    r"\checkpoints\best_model_v6.pth"
)

OUTPUT_DIR = (
    r"C:\KLA\results\model_v6\outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_NAME = "003184.npy"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("V6 PREDICTION")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
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

print("V6 checkpoint loaded successfully.")

print(
    "Checkpoint PSNR:",
    checkpoint.get("psnr", "N/A")
)

print(
    "Checkpoint SSIM:",
    checkpoint.get("ssim", "N/A")
)


# ============================================================
# LOAD INPUT
# ============================================================

input_path = os.path.join(
    INPUT_DIR,
    IMAGE_NAME
)

gt_path = os.path.join(
    GT_DIR,
    IMAGE_NAME
)

noisy = np.load(
    input_path
).astype(np.float32)

gt = np.load(
    gt_path
).astype(np.float32)


print()
print("Input:", IMAGE_NAME)

print(
    "NoisyLR shape:",
    noisy.shape
)

print(
    "GT shape:",
    gt.shape
)


# ============================================================
# NORMALIZE INPUT
# ============================================================

# NoisyLR is 128x128 and GT is 256x256.
# Convert input to tensor.

input_tensor = torch.from_numpy(
    noisy
).unsqueeze(0).unsqueeze(0)

input_tensor = input_tensor.to(
    device
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("Running V6 prediction...")

with torch.no_grad():

    output = model(
        input_tensor
    )

output = output.squeeze().cpu().numpy()

output = np.clip(
    output,
    0.0,
    1.0
)


print(
    "Output shape:",
    output.shape
)


# ============================================================
# SAVE NUMPY OUTPUT
# ============================================================

output_npy = os.path.join(
    OUTPUT_DIR,
    "003184_v6_output.npy"
)

np.save(
    output_npy,
    output
)

print(
    "Saved NPY:",
    output_npy
)


# ============================================================
# SAVE PNG IMAGES
# ============================================================

noisy_display = noisy.copy()

# Normalize only for visualization
noisy_min = noisy_display.min()
noisy_max = noisy_display.max()

if noisy_max > noisy_min:

    noisy_display = (
        noisy_display - noisy_min
    ) / (
        noisy_max - noisy_min
    )

else:

    noisy_display = np.zeros_like(
        noisy_display
    )


# Save restored image
output_png = os.path.join(
    OUTPUT_DIR,
    "003184_v6_output.png"
)

plt.imsave(
    output_png,
    output,
    cmap="gray",
    vmin=0,
    vmax=1
)


# Save GT
gt_png = os.path.join(
    OUTPUT_DIR,
    "003184_GT.png"
)

plt.imsave(
    gt_png,
    np.clip(gt, 0, 1),
    cmap="gray",
    vmin=0,
    vmax=1
)


# Save noisy input
noisy_png = os.path.join(
    OUTPUT_DIR,
    "003184_NoisyLR.png"
)

plt.imsave(
    noisy_png,
    noisy_display,
    cmap="gray",
    vmin=0,
    vmax=1
)


# ============================================================
# COMPARISON IMAGE
# ============================================================

comparison_png = os.path.join(
    OUTPUT_DIR,
    "003184_comparison_v6.png"
)

plt.figure(
    figsize=(15, 5)
)

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    noisy_display,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "NoisyLR"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    output,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "V6 Restored"
)

plt.axis("off")


plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    np.clip(gt, 0, 1),
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Ground Truth"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    comparison_png,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# METRICS
# ============================================================

pred_tensor = torch.from_numpy(
    output
).float().unsqueeze(0).unsqueeze(0)

gt_tensor = torch.from_numpy(
    gt
).float().unsqueeze(0).unsqueeze(0)


mse = torch.mean(
    (pred_tensor - gt_tensor) ** 2
).item()

psnr = 10.0 * np.log10(
    1.0 / max(mse, 1e-10)
)


print()
print("=" * 60)
print("V6 OUTPUT COMPLETE")
print("=" * 60)

print(
    "PSNR:",
    f"{psnr:.4f} dB"
)

print(
    "MSE:",
    f"{mse:.8f}"
)

print()
print(
    "Output folder:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Restored image:"
)

print(
    output_png
)

print()
print(
    "Comparison image:"
)

print(
    comparison_png
)

print("=" * 60)