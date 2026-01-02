# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 18:29:44 2025

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
image_path = r"C:\Users\User\OneDrive\Pictures\Strawberry.jpeg"
img = Image.open(image_path)
img_np = np.array(img)

# =========================
# 2. RGB → HSV
# =========================
hsv = mcolors.rgb_to_hsv(img_np / 255.0)
H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# =========================
# 3. HSV Threshold (RED)
# =========================
binary = (
    ((H < 0.05) | (H > 0.95)) &
    (S > 0.45) &
    (V > 0.3)
)

# =========================
# 4. Morphology
# =========================
binary = ndimage.binary_closing(binary, structure=np.ones((9,9)))
binary = ndimage.binary_fill_holes(binary)

# =========================
# 5. Distance Transform
# =========================
distance = ndimage.distance_transform_edt(binary)

# =========================
# 6. Local Maxima (Markers)
# =========================
local_max = ndimage.maximum_filter(distance, size=40) == distance
markers, _ = ndimage.label(local_max)

# =========================
# 7. Watershed-style Separation
# =========================
labels = ndimage.watershed_ift(
    (255 - (distance / distance.max() * 255)).astype(np.uint8),
    markers
)

# Mask background
labels[binary == 0] = 0

# =========================
# 8. Count Objects
# =========================
unique_labels = np.unique(labels)
count = len(unique_labels) - 1  # remove background

print("Final Strawberry Count:", count)

# =========================
# 9. Display
# =========================
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(binary, cmap="gray")
plt.title("HSV Segmentation")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(distance, cmap="inferno")
plt.title("Distance Transform")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(labels, cmap="nipy_spectral")
plt.title(f"Detected Strawberry = {count}")
plt.axis("off")

plt.tight_layout()
plt.show()
