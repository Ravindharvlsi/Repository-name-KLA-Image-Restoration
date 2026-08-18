import os
import sys
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

sys.path.insert(0, r"C:\KLA\code")

from dataset import ImageRestorationDataset
from model_v5 import V5RestorationNet


GT_DIR = r"C:\KLA\train\train\GT"
NOISY_DIR = r"C:\KLA\train\train\NoisyLR"

CHECKPOINT_DIR = (
    r"C:\KLA\results\model_v5\checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# =========================
# SETTINGS
# =========================

SEED = 42

EPOCHS = 15

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

MIN_LR = 1e-6

NUM_WORKERS = 0


# =========================
# SEED
# =========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("V5 IMAGE RESTORATION TRAINING")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory / (1024 ** 3),
            2
        ),
        "GB"
    )

print("=" * 70)


# =========================
# DATASET
# =========================

files = sorted([
    f
    for f in os.listdir(GT_DIR)
    if f.endswith(".npy")
])

dataset = ImageRestorationDataset(
    GT_DIR,
    NOISY_DIR,
    files
)

print(
    "Total samples:",
    len(dataset)
)


# =========================
# SPLIT
# =========================

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset) - train_size
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator()
    .manual_seed(SEED)
)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# =========================
# LOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# =========================
# MODEL
# =========================

model = V5RestorationNet(
    in_channels=1,
    out_channels=1,
    features=80,
    num_blocks=24
).to(device)

print("Model: V5RestorationNet")
print("Features: 80")
print("Residual blocks: 24")


# =========================
# LOSSES
# =========================

mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


def charbonnier_loss(pred, target):

    eps = 1e-3

    diff = pred - target

    return torch.sqrt(
        diff * diff + eps * eps
    ).mean()


def ssim_loss(pred, target):

    pred = torch.clamp(
        pred, 0.0, 1.0
    )

    target = torch.clamp(
        target, 0.0, 1.0
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(
        pred, 7, 1, 3
    )

    mu_y = F.avg_pool2d(
        target, 7, 1, 3
    )

    sigma_x = (
        F.avg_pool2d(
            pred * pred,
            7, 1, 3
        )
        - mu_x * mu_x
    )

    sigma_y = (
        F.avg_pool2d(
            target * target,
            7, 1, 3
        )
        - mu_y * mu_y
    )

    sigma_xy = (
        F.avg_pool2d(
            pred * target,
            7, 1, 3
        )
        - mu_x * mu_y
    )

    num = (
        (2 * mu_x * mu_y + c1)
        *
        (2 * sigma_xy + c2)
    )

    den = (
        (mu_x * mu_x +
         mu_y * mu_y + c1)
        *
        (sigma_x +
         sigma_y + c2)
    )

    score = num / (
        den + 1e-8
    )

    return 1.0 - score.mean()


def combined_loss(pred, target):

    charbonnier = charbonnier_loss(
        pred,
        target
    )

    mse = mse_loss(
        pred,
        target
    )

    l1 = l1_loss(
        pred,
        target
    )

    ssim = ssim_loss(
        pred,
        target
    )

    return (
        0.55 * charbonnier
        + 0.20 * mse
        + 0.15 * l1
        + 0.10 * ssim
    )


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)


# =========================
# SCHEDULER
# =========================

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LR
    )
)


# =========================
# AMP
# =========================

use_amp = torch.cuda.is_available()

scaler = torch.cuda.amp.GradScaler(
    enabled=use_amp
)


# =========================
# METRICS
# =========================

def calculate_psnr(pred, target):

    pred = torch.clamp(
        pred, 0.0, 1.0
    )

    target = torch.clamp(
        target, 0.0, 1.0
    )

    mse = torch.mean(
        (pred - target) ** 2,
        dim=(1, 2, 3)
    )

    mse = torch.clamp(
        mse,
        min=1e-10
    )

    psnr = 10.0 * torch.log10(
        1.0 / mse
    )

    return psnr.mean().item()


def calculate_ssim(pred, target):

    pred = torch.clamp(
        pred, 0.0, 1.0
    )

    target = torch.clamp(
        target, 0.0, 1.0
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(
        pred, 7, 1, 3
    )

    mu_y = F.avg_pool2d(
        target, 7, 1, 3
    )

    sigma_x = (
        F.avg_pool2d(
            pred * pred,
            7, 1, 3
        )
        - mu_x * mu_x
    )

    sigma_y = (
        F.avg_pool2d(
            target * target,
            7, 1, 3
        )
        - mu_y * mu_y
    )

    sigma_xy = (
        F.avg_pool2d(
            pred * target,
            7, 1, 3
        )
        - mu_x * mu_y
    )

    num = (
        (2 * mu_x * mu_y + c1)
        *
        (2 * sigma_xy + c2)
    )

    den = (
        (mu_x * mu_x +
         mu_y * mu_y + c1)
        *
        (sigma_x +
         sigma_y + c2)
    )

    return (
        num / (den + 1e-8)
    ).mean().item()


# =========================
# TRAINING
# =========================

best_psnr = -float("inf")
best_ssim = -float("inf")

total_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1
):

    start = time.time()

    model.train()

    train_loss = 0.0

    for noisy, gt in train_loader:

        noisy = noisy.to(
            device,
            non_blocking=True
        )

        gt = gt.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):

            output = model(noisy)

            loss = combined_loss(
                output,
                gt
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        train_loss += loss.item()


    train_loss /= len(train_loader)


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    val_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():

        for noisy, gt in val_loader:

            noisy = noisy.to(
                device,
                non_blocking=True
            )

            gt = gt.to(
                device,
                non_blocking=True
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):

                output = model(noisy)

                loss = combined_loss(
                    output,
                    gt
                )

            val_loss += loss.item()

            total_psnr += calculate_psnr(
                output.float(),
                gt.float()
            )

            total_ssim += calculate_ssim(
                output.float(),
                gt.float()
            )


    val_loss /= len(val_loader)

    epoch_psnr = (
        total_psnr /
        len(val_loader)
    )

    epoch_ssim = (
        total_ssim /
        len(val_loader)
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    scheduler.step()

    elapsed = (
        time.time() - start
    ) / 60.0


    # =========================
    # PRINT
    # =========================

    print()
    print("=" * 70)

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss : {train_loss:.6f}"
    )

    print(
        f"Val Loss   : {val_loss:.6f}"
    )

    print(
        f"PSNR       : {epoch_psnr:.4f} dB"
    )

    print(
        f"SSIM       : {epoch_ssim:.6f}"
    )

    print(
        f"LR         : {current_lr:.8f}"
    )

    print(
        f"Time       : {elapsed:.2f} min"
    )

    print("=" * 70)


    # =========================
    # SAVE BEST
    # =========================

    if epoch_psnr > best_psnr:

        best_psnr = epoch_psnr
        best_ssim = epoch_ssim

        checkpoint = os.path.join(
            CHECKPOINT_DIR,
            "best_model_v5.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_loss": val_loss,
                "psnr": epoch_psnr,
                "ssim": epoch_ssim
            },
            checkpoint
        )

        print()
        print("NEW BEST V5 MODEL SAVED!")

        print(
            "PSNR:",
            f"{epoch_psnr:.4f} dB"
        )

        print(
            "SSIM:",
            f"{epoch_ssim:.6f}"
        )

        print(
            "Checkpoint:",
            checkpoint
        )


# =========================
# COMPLETE
# =========================

total_time = (
    time.time() - total_start
) / 60.0

print()
print("=" * 70)
print("V5 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best PSNR: {best_psnr:.6f} dB"
)

print(
    f"Best SSIM: {best_ssim:.6f}"
)

print(
    "Checkpoint:",
    os.path.join(
        CHECKPOINT_DIR,
        "best_model_v5.pth"
    )
)

print(
    f"Total Training Time: {total_time:.2f} min"
)

print("=" * 70)
