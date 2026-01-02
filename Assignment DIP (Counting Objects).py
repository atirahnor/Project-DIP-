# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 18:12:11 2025

@author: User
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.colors as mcolors

# =========================
# 1. Load Image
# =========================
image_path = r"C:\Users\User\OneDrive\Pictures\Takoyaki.jpeg"
img = Image.open(image_path)
img_np = np.array(img)

# =========================
# 2. Convert RGB to HSV
# =========================
img_hsv = mcolors.rgb_to_hsv(img_np / 255.0)

H = img_hsv[:, :, 0]
S = img_hsv[:, :, 1]
V = img_hsv[:, :, 2]

# =========================
# 3. HSV Thresholding
# (yellow-brown takoyaki)
# =========================
binary = (
    (H > 0.05) & (H < 0.15) &   # Hue range
    (S > 0.35) &               # remove white plate
    (V > 0.3)
)

# =========================
# 4. Morphological Processing
# =========================
binary = ndimage.binary_opening(binary, structure=np.ones((5,5)))
binary = ndimage.binary_closing(binary, structure=np.ones((9,9)))
binary = ndimage.binary_fill_holes(binary)

# =========================
# 5. Connected Component Labeling
# =========================
labels, num_labels = ndimage.label(binary)

# =========================
# 6. Area Filtering (Remove Noise)
# =========================
sizes = ndimage.sum(binary, labels, range(1, num_labels + 1))

min_size = 3000   # threshold area for takoyaki
cleaned = np.zeros_like(binary)

count = 0
for i, size in enumerate(sizes):
    if size > min_size:
        cleaned[labels == (i + 1)] = 1
        count += 1

print("Final Takoyaki Count:", count)

# =========================
# 7. Display Results
# =========================
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(binary, cmap="gray")
plt.title("HSV Segmentation (Before Filtering)")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(cleaned, cmap="gray")
plt.title("After Area Filtering")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(cleaned, cmap="nipy_spectral")
plt.title(f"Detected Takoyaki = {count}")
plt.axis("off")

plt.tight_layout()
plt.show()
