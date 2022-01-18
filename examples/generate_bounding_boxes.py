import os
from csdlib import CSDHandler
import numpy as np

dataset_path = "../../CYGNO-ML-DATASET"
padding = 10

# Create an instance of the dataset handler providing the path
dataset = CSDHandler(dataset_path)

# Fetch all available experiment ids
exp_list = dataset.list_all_experiments()

# Initialize an empty rows string
rows = ""

# Iterate all experiments to generate fill rows
for experiment_id in exp_list:

    image_name = f"{experiment_id}.png" # TODO: get it from the description

    # Load particles
    particles, p_types = dataset.load_experiment_trajs(experiment_id)

    # Generate particles boxes
    boxes = ''
    for i, p in enumerate(particles):
        x = p[:, 0]
        y = p[:, 1]
        xmin = max(0, int(np.min(x) * 2304/350) - padding)    #TODO: Read this from metadata
        ymin = max(0, int(np.min(y) * 2304/350) - padding)    #TODO: Read this from metadata
        xmax = min(int(np.max(x) * 2304/350) + padding, 2304) #TODO: Read this from metadata
        ymax = min(int(np.max(y) * 2304/350) + padding, 2304) #TODO: Read this from metadata
        class_id = p_types[i]
        box = f" {xmin},{ymin},{xmax},{ymax},{class_id}"
        boxes += box 

    rows += f"{image_name}{boxes}\n"

print(rows)


with open(os.path.join(dataset_path, "boxes.txt"), "w") as box_file:
    box_file.write(rows)

with open(os.path.join(dataset_path, "particle_classes.txt"), "w") as box_file:
    box_file.write('ER\nNR')


