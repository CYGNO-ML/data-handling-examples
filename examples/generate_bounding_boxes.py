import os
from cygnods import CygnoDataset
from cygnods.annotators import DatasetAnnotator
import numpy as np

dataset_path = "../CYGNO-ML-DATASET"

# Create an instance of the dataset handler providing the path
dataset = CygnoDataset(dataset_path)

# Create an instance of the dataset annotator
annotator = DatasetAnnotator(dataset)

# Annotate and create tran/test/val split
annotator.create_yolov5_annotations(0.8, 0.1, 0.1)