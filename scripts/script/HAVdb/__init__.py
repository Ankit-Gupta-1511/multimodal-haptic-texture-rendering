# This file is intentionally left blank
import matplotlib.pyplot as plt
import numpy as np
import sys

import os
import json



angles = [-90, 0, 90, 180]
# color = ['b', 'g', 'r', 'y']
# color = [[0, 0, 1], [0, 0.5, 0], [1, 0, 0], [0.8, 0.8, 0]]
markers = ['o', 'v', 's', '^']

sampling_rate = 12000

#path to the data folder as the first argument
path = "../datah5/"
users=["subject_0", "subject_1", "subject_2", "subject_3", "subject_4", "subject_5", "subject_6", "subject_7", "subject_8", "subject_9"]
textures = ["7t", "10t", "18t", "39t", "44t", "54t", "59t", "83t", "108t", "120t"]
trials=["std_0", "std_1", "std_2", "cte_force_3", "cte_speed_4"]
streams=["kistler","ft_sensor","positions"]

def get_texture_metadata():
    return json.load(open(path + "texture_metadata.json"))