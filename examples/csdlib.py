import os
import cv2
import json
import numpy as np


class CSDHandler():
    def __init__(self, path):
        #TODO: Check if exists
        self.path = path
    
    def load_experiment(self, experiment_id):
        trajs, t_classes = self.load_experiment_trajs(experiment_id=experiment_id)
        cmos = self.load_experiment_cmos(experiment_id=experiment_id)
        pmt = self.load_experiment_pmt(experiment_id=experiment_id)
        return trajs, t_classes, cmos, pmt
    
    def load_experiment_description(self, experiment_id):
        experiment_path = os.path.join(self.path, experiment_id)
        desc_path = os.path.join(experiment_path, "description.json")
        with open(desc_path, encoding="utf-8") as json_file:
            desc = json.load(json_file)
        return desc
    
    def load_experiment_trajs(self, experiment_id):
        experiment_path = os.path.join(self.path, experiment_id)
        desc = self.load_experiment_description(experiment_id)
        particles, p_types = [], []
        for p in desc["particles_info"]:
            part_path = os.path.join(experiment_path, p["file"])
            particles.append(np.loadtxt(part_path))
            p_types.append(p["type"])
        return particles, p_types
    
    def load_experiment_cmos(self, experiment_id):
        experiment_path = os.path.join(self.path, experiment_id)
        image_path = os.path.join(experiment_path, "img.png")
        return cv2.imread(image_path)
    
    def load_experiment_pmt(self, experiment_id):
        experiment_path = os.path.join(self.path, experiment_id)
        pmt_path = os.path.join(experiment_path, "pmt.npy")
        return np.load(pmt_path)

    def load_all_trajs(self):
        experiment_ids = os.listdir(self.path)
        p, p_type = [], []
        for e_id in experiment_ids:
            pi, pti = self.load_experiment_trajs(e_id)
            p += pi
            p_type += pti
        return p, p_type

    def load_all_cmos(self):
        experiment_ids = os.listdir(self.path)
        all_cmos = []
        for e_id in experiment_ids:
            c_cmos = self.load_experiment_cmos(e_id)
            all_cmos.append(c_cmos)
        return all_cmos

    def load_all_pmt(self):
        experiment_ids = os.listdir(self.path)
        all_pmt = []
        for e_id in experiment_ids:
            c_pmt = self.load_experiment_pmt(e_id)
            all_pmt.append(c_pmt)
        return all_pmt