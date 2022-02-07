from cygnods import CygnoDataset
import matplotlib.pyplot as plt

dataset_path = "../../CYGNO-ML-DATASET"
experiment_id = "0011"
flip = True


# Create an instance of the dataset handler providing the path
dataset = CygnoDataset(dataset_path)

# Load image
image = dataset.load_experiment_cmos(experiment_id)

# Load PMT
pmt = dataset.load_experiment_pmt(experiment_id)

# Load particles
particles, p_types = dataset.load_experiment_trajs(experiment_id)

# Plot particles
fig = plt.figure(figsize=(3, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[0.45, 0.45, 0.1])

ax1 = fig.add_subplot(gs[0, 0], projection="3d")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

for p in particles:
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    en = p[:, 3]
    t = p[:, 4]
    ax1.plot(xs=x, ys=y, zs=z)


ax1.set_xlim3d(0, 350) # First dimension of the detector [mm]
ax1.set_ylim3d(0, 350) # Second dimension of the detector [mm]
ax1.set_zlim3d(0, 510) # Third dimension of the detector [mm]

ax1.set_title("Trajectories of particles inside the TPC detector")

# Plot image
ax2 = fig.add_subplot(gs[1, 0])
origin = 'lower' if flip else 'upper'
ax2.imshow(image, origin=origin)
ax2.set_title("CMOS")

# Plot pmt
ax3 = fig.add_subplot(gs[2, 0])
if pmt is not None:
    ax3.plot(pmt[::10000])
ax3.set_title("PMT")

fig.tight_layout()
plt.show()
