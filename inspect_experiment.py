import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


dataset_path = "../dataset-generator/output"
experiment_id = "0001"

experiment_path = os.path.join(dataset_path, experiment_id)

# Load description
desc_path = os.path.join(experiment_path, "description.json")
with open(desc_path, encoding="utf-8") as json_file:
    desc = json.load(json_file)

# Load image
image_path = os.path.join(experiment_path, "img.png")
image = cv2.imread(image_path)

# Load PMT
pmt_path = os.path.join(experiment_path, "pmt.npy")
pmt = np.load(pmt_path)

# Load particles
particles, p_types = [], []
for p in desc["particles_info"]:
    part_path = os.path.join(experiment_path, p["file"])
    particles.append(np.loadtxt(part_path))
    p_types.append(p["type"])

# Plot particles
fig = plt.figure(figsize=(3, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[0.45, 0.45, 0.1])

ax1 = fig.add_subplot(gs[0, 0], projection="3d")
for p in particles:
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    en = p[:, 3]
    t = p[:, 4]
    ax1.plot(xs=x, ys=y, zs=z)

x_dim = 350           # First dimension of the detector [mm]
y_dim = 350           # Second dimension of the detector [mm]
z_dim = 510           # Third dimension of the detector [mm]

ax1.set_xlim3d(0, x_dim)
ax1.set_ylim3d(0, y_dim)
ax1.set_zlim3d(0, z_dim)
ax1.set_title("Box")

# Plot image
ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(image)
ax2.set_title("CMOS")

# Plot pmt
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(pmt[::10000])
ax3.set_title("PMT")

fig.tight_layout()
plt.show()
