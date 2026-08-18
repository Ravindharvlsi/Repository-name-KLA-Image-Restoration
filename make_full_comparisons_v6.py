import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lpips

from skimage.metrics import structural_similarity


# ============================================================
# PATHS
# ============================================================

NOISY_DIR = r"C:\KLA\train\train\NoisyLR"
GT_DIR = r"C:\KLA\train\train\GT"

CHECKPOINT = (
    r"C:\KLA\results\model_v6\checkpoints\best_model_v6.pth"
)

OUTPUT_DIR = (
    r"C:\KLA\results\model_v6\outputs\full_comparisons"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

import sys
sys.path.insert(0, r"C:\KLA\code")

from model_v5 import V5RestorationNet


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("V6 FULL COMPARISON GENERATION")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD V6
# ============================================================

model = V5RestorationNet(
    in_channels=1,
    out_channels=1,
    features=80,
    num_blocks=24
).to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "V6 checkpoint loaded."
)

print(
    "Training Best PSNR:",
    checkpoint.get("psnr", "N/A")
)

print(
    "Training Best SSIM:",
    checkpoint.get("ssim", "N/A")
)


# ============================================================
# LPIPS
# ============================================================

print()
print("Loading LPIPS model...")

lpips_model = lpips.LPIPS(
    net="alex"
).to(device)

lpips_model.eval()

print("LPIPS loaded.")


# ============================================================
# FILES
# ============================================================

files = sorted([
    f for f in os.listdir(NOISY_DIR)
    if f.endswith(".npy")
])

print()
print(
    "Total images:",
    len(files)
)

print()


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def psnr(pred, gt):

    mse = np.mean(
        (pred - gt) ** 2
    )

    if mse < 1e-12:
        return 99.0

    return 10.0 * np.log10(
        1.0 / mse
    )


def ssim(pred, gt):

    return structural_similarity(
        gt,
        pred,
        data_range=1.0
    )


def make_lpips_tensor(image):

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    # LPIPS expects 3 channels and [-1, 1]
    tensor = tensor.repeat(
        1, 3, 1, 1
    )

    tensor = tensor * 2.0 - 1.0

    return tensor.to(device)


# ============================================================
# PROCESS ALL
# ============================================================

with torch.no_grad():

    for index, filename in enumerate(
        files,
        1
    ):

        name = os.path.splitext(
            filename
        )[0]

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        noisy = np.load(
            os.path.join(
                NOISY_DIR,
                filename
            )
        ).astype(np.float32)

        gt = np.load(
            os.path.join(
                GT_DIR,
                filename
            )
        ).astype(np.float32)

        gt = np.clip(
            gt,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Normalize noisy for display
        # ----------------------------------------------------

        noisy_min = noisy.min()
        noisy_max = noisy.max()

        if noisy_max > noisy_min:

            noisy_display = (
                noisy - noisy_min
            ) / (
                noisy_max - noisy_min
            )

        else:

            noisy_display = np.zeros_like(
                noisy
            )

        noisy_display = np.clip(
            noisy_display,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Bicubic
        # ----------------------------------------------------

        noisy_tensor = torch.from_numpy(
            noisy_display
        ).float()

        noisy_tensor = (
            noisy_tensor
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        bicubic_tensor = F.interpolate(
            noisy_tensor,
            size=gt.shape,
            mode="bicubic",
            align_corners=False
        )

        bicubic = (
            bicubic_tensor
            .squeeze()
            .cpu()
            .numpy()
        )

        bicubic = np.clip(
            bicubic,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # V6 AI restoration
        # ----------------------------------------------------

        raw_noisy = torch.from_numpy(
            noisy
        ).float()

        raw_noisy = (
            raw_noisy
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        restored = model(
            raw_noisy
        )

        restored = (
            restored
            .squeeze()
            .cpu()
            .numpy()
        )

        restored = np.clip(
            restored,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Make sure size matches GT
        # ----------------------------------------------------

        if restored.shape != gt.shape:

            restored_tensor = torch.from_numpy(
                restored
            ).float()

            restored_tensor = (
                restored_tensor
                .unsqueeze(0)
                .unsqueeze(0)
            )

            restored = F.interpolate(
                restored_tensor,
                size=gt.shape,
                mode="bicubic",
                align_corners=False
            ).squeeze().numpy()

            restored = np.clip(
                restored,
                0.0,
                1.0
            )

        # ====================================================
        # METRICS
        # ====================================================

        bicubic_psnr = psnr(
            bicubic,
            gt
        )

        bicubic_ssim = ssim(
            bicubic,
            gt
        )

        restored_psnr = psnr(
            restored,
            gt
        )

        restored_ssim = ssim(
            restored,
            gt
        )

        # LPIPS
        gt_lpips = make_lpips_tensor(
            gt
        )

        bicubic_lpips = lpips_model(
            make_lpips_tensor(bicubic),
            gt_lpips
        ).item()

        restored_lpips = lpips_model(
            make_lpips_tensor(restored),
            gt_lpips
        ).item()

        # ====================================================
        # FIGURE
        # ====================================================

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(20, 5)
        )

        # ----------------------------------------------------
        # NoisyLR
        # ----------------------------------------------------

        axes[0].imshow(
            noisy_display,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0].set_title(
            f"NoisyLR\n"
            f"({noisy.shape[0]}x{noisy.shape[1]})"
        )

        axes[0].axis("off")

        # ----------------------------------------------------
        # Bicubic
        # ----------------------------------------------------

        axes[1].imshow(
            bicubic,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1].set_title(
            f"Bicubic\n"
            f"PSNR: {bicubic_psnr:.2f} dB\n"
            f"SSIM: {bicubic_ssim:.4f}\n"
            f"LPIPS: {bicubic_lpips:.4f}"
        )

        axes[1].axis("off")

        # ----------------------------------------------------
        # V6
        # ----------------------------------------------------

        axes[2].imshow(
            restored,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[2].set_title(
            f"V6 AI Restored\n"
            f"PSNR: {restored_psnr:.2f} dB\n"
            f"SSIM: {restored_ssim:.4f}\n"
            f"LPIPS: {restored_lpips:.4f}"
        )

        axes[2].axis("off")

        # ----------------------------------------------------
        # Ground Truth
        # ----------------------------------------------------

        axes[3].imshow(
            gt,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[3].set_title(
            f"Ground Truth\n"
            f"({gt.shape[0]}x{gt.shape[1]})"
        )

        axes[3].axis("off")

        # ----------------------------------------------------
        # Main title
        # ----------------------------------------------------

        fig.suptitle(
            f"Comparison: {filename}",
            fontsize=16,
            fontweight="bold"
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            name + "_comparison.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            index % 10 == 0
            or index == 1
            or index == len(files)
        ):

            print(
                f"[{index}/{len(files)}] "
                f"{filename} | "
                f"V6 PSNR: {restored_psnr:.2f} dB | "
                f"SSIM: {restored_ssim:.4f} | "
                f"LPIPS: {restored_lpips:.4f}"
            )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("ALL COMPARISON IMAGES CREATED")
print("=" * 70)

print(
    "Total:",
    len(files)
)

print(
    "Saved to:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Open folder:"
)

print(
    "explorer " + OUTPUT_DIR
)

print("=" * 70)