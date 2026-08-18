import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, r"C:\KLA\code")

from dataset import ImageRestorationDataset
from model import ImageRestorationNet


# ============================================================
# PATHS
# ============================================================

GT_DIR = r"C:\KLA\train\train\GT"
NOISY_DIR = r"C:\KLA\train\train\NoisyLR"

RESULTS_DIR = r"C:\KLA\results"
CHECKPOINT_DIR = r"C:\KLA\results\checkpoints"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 8
EPOCHS = 100

INITIAL_LR = 1e-4
MIN_LR = 1e-6


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE       :", device)

if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))
    print(
        "GPU MEMORY   :",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

print("=" * 60)


# ============================================================
# DATASET
# ============================================================

files = sorted([
    f for f in os.listdir(GT_DIR)
    if f.endswith(".npy")
])

dataset = ImageRestorationDataset(
    GT_DIR,
    NOISY_DIR,
    files
)

print("TOTAL SAMPLES:", len(dataset))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print("TRAIN SAMPLES :", len(train_dataset))
print("VAL SAMPLES   :", len(val_dataset))


# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = ImageRestorationNet().to(device)

print("MODEL LOADED")


# ============================================================
# LOSS
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=INITIAL_LR
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=MIN_LR
)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(pred, target):

    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    mse = torch.mean(
        (pred - target) ** 2
    ).item()

    if mse == 0:
        return 100.0

    return 10.0 * np.log10(1.0 / mse)


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(pred, target):

    pred = torch.clamp(
        pred,
        0.0,
        1.0
    ).detach().cpu().numpy()

    target = torch.clamp(
        target,
        0.0,
        1.0
    ).detach().cpu().numpy()

    scores = []

    for i in range(pred.shape[0]):

        score = ssim(
            target[i, 0],
            pred[i, 0],
            data_range=1.0
        )

        scores.append(score)

    return float(np.mean(scores))


# ============================================================
# TRAINING
# ============================================================

best_psnr = -float("inf")

total_start = time.time()


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()


    # ========================================================
    # TRAIN
    # ========================================================

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

        output = model(noisy)

        loss = criterion(
            output,
            gt
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0
    psnr_total = 0.0
    ssim_total = 0.0

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

            output = model(noisy)

            loss = criterion(
                output,
                gt
            )

            val_loss += loss.item()

            psnr_total += calculate_psnr(
                output,
                gt
            )

            ssim_total += calculate_ssim(
                output,
                gt
            )


    val_loss /= len(val_loader)
    psnr_value = psnr_total / len(val_loader)
    ssim_value = ssim_total / len(val_loader)


    # ========================================================
    # LEARNING RATE
    # ========================================================

    current_lr = optimizer.param_groups[0]["lr"]

    scheduler.step()


    # ========================================================
    # TIME
    # ========================================================

    epoch_time = (
        time.time() - epoch_start
    ) / 60.0


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 60)

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
        f"PSNR       : {psnr_value:.4f} dB"
    )

    print(
        f"SSIM       : {ssim_value:.6f}"
    )

    print(
        f"LR         : {current_lr:.8f}"
    )

    print(
        f"Time       : {epoch_time:.2f} min"
    )

    print("=" * 60)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if psnr_value > best_psnr:

        best_psnr = psnr_value

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "best_psnr_model.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "psnr": psnr_value,
                "ssim": ssim_value
            },
            checkpoint_path
        )

        print()
        print("🏆 NEW BEST MODEL SAVED!")
        print(
            f"PSNR = {psnr_value:.4f} dB"
        )
        print()


# ============================================================
# COMPLETE
# ============================================================

total_time = (
    time.time() - total_start
) / 60.0

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best PSNR: {best_psnr:.6f} dB"
)

print(
    "Checkpoint:",
    os.path.join(
        CHECKPOINT_DIR,
        "best_psnr_model.pth"
    )
)

print(
    f"Total Training Time: {total_time:.2f} min"
)

print("=" * 60)