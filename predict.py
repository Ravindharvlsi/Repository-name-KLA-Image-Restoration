import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r"C:\KLA\code")

from model import ImageRestorationNet


MODEL_PATH = r"C:\KLA\results\checkpoints\best_psnr_model.pth"
INPUT_PATH = r"C:\KLA\train\train\NoisyLR\003184.npy"
OUTPUT_PATH = r"C:\KLA\results\restored\003184_restored.png"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = ImageRestorationNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

image = np.load(INPUT_PATH).astype(np.float32)

tensor = torch.from_numpy(image)
tensor = tensor.unsqueeze(0).unsqueeze(0)
tensor = tensor.to(device)

with torch.no_grad():
    output = model(tensor)

output = torch.clamp(output, 0.0, 1.0)

output = output.squeeze().cpu().numpy()

output = (output * 255).astype(np.uint8)

Image.fromarray(output).save(OUTPUT_PATH)

print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)
print("Restored image saved successfully!")