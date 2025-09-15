##############################################################################################
## main.py                                                                                  ##
## The first thing I'll do as part of the implementation is to replicate this notebook:     ##
## https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing    ##
## into this file. I will disect it part by part, and integrate MISO into it eventually.    ##
## This means that this main file will be a bit long and messy, but I will slowly shorten   ##
## it down.                                                                                 ##
## ---------------------------------------------------------------------------------------- ##
## Author:   Mathias Otnes                                                                  ##
## Date:     09/15/2025                                                                     ##
##############################################################################################



#########################################################
##                    Environment                      ##
#########################################################

import numpy as np
np.bool = np.bool_ # bad trick to fix numpy version issue :(
import os
import sys
import shutil
import pickle

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

# Force Python to prefer peract_colab/rlbench (dev mode) instead of mod/RLBench
colab_rlbench_path = os.path.join(os.path.dirname(__file__), "peract_colab", "rlbench")
if os.path.isdir(colab_rlbench_path):
    sys.path.insert(0, os.path.dirname(colab_rlbench_path))
    print("[Info] Using rlbench from peract_colab (dev mode).")
else:
    print("[Info] Using rlbench from mod/RLBench (full install).")

from rlbench.utils import get_stored_demo
from rlbench.backend.utils import extract_obs

# These needs to be defined when using Linux
# os.environ["DISPLAY"] = ":0"
# os.environ["PYOPENGL_PLATFORM"] = "egl"


#########################################################
##                   Configuration                     ##
#########################################################

# constants
TASK                        = 'open_drawer'
DATA_FOLDER                 = 'peract_colab/data'
EPISODES_FOLDER             = 'colab_dataset/open_drawer/all_variations/episodes'
EPISODE_FOLDER              = 'episode%d'
CAMERAS                     = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
LOW_DIM_SIZE                = 4     # {left_finger_joint, right_finger_joint, gripper_open, timestep}
IMAGE_SIZE                  = 128   # 128x128 - if you want to use higher voxel resolutions like 200^3, you might want to regenerate the dataset with larger images
VARIATION_DESCRIPTIONS_PKL  = 'variation_descriptions.pkl' # the pkl file that contains language goals for each demonstration
EPISODE_LENGTH              = 10    # max steps for agents
DEMO_AUGMENTATION_EVERY_N   = 10    # sample n-th frame in demo
ROTATION_RESOLUTION         = 5     # degree increments per axis

# settings
VOXEL_SIZES     = [100] # 100x100x100 voxels
NUM_LATENTS     = 512   # PerceiverIO latents
SCENE_BOUNDS    = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6] # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
BATCH_SIZE      = 1
NUM_DEMOS       = 8     # total number of training demonstrations to use while training PerAct
NUM_TEST        = 2     # episodes to evaluate on


#########################################################
##                      Storage                        ##
#########################################################

# Storing replay buffer to disc instead of RAM
sys.path.append('peract_colab')
data_path = os.path.join(DATA_FOLDER, EPISODES_FOLDER)

train_replay_storage_dir = 'replay_train'
if not os.path.exists(train_replay_storage_dir):
  os.mkdir(train_replay_storage_dir)

test_replay_storage_dir = 'replay_test'
if not os.path.exists(test_replay_storage_dir):
  os.mkdir(test_replay_storage_dir)


#########################################################
##                    Data Loading                     ##
#########################################################


def visualize_demo() -> None:
    """ Visualizes a demo from a pre-generated dataset. """

    # what to visualize
    episode_idx_to_visualize = 1 # out of 10 demos
    ts = 50 # timestep out of total timesteps

    # get demo
    demo = get_stored_demo(data_path=data_path, index=episode_idx_to_visualize)

    # extract obs at timestep
    obs_dict = extract_obs(demo._observations[ts], CAMERAS, t=ts)

    # total timesteps in demo
    print(f"Demo {episode_idx_to_visualize} | {len(demo._observations)} total steps\n")

    # plot rgb and depth at timestep
    fig = plt.figure(figsize=(20, 10))
    rows, cols = 2, len(CAMERAS)

    plot_idx = 1
    for camera in CAMERAS:
        # rgb
        rgb_name = f"{camera}_rgb"
        rgb = np.array(obs_dict[rgb_name], copy=True)
        if rgb.ndim == 3 and rgb.shape[0] in (3, 4):  # (C, H, W) → (H, W, C)
            rgb = np.transpose(rgb, (1, 2, 0))
        fig.add_subplot(rows, cols, plot_idx)
        plt.imshow(rgb)
        plt.axis('off')
        plt.title(f"{camera}_rgb | step {ts}")

        # depth
        depth_name = f"{camera}_depth"
        depth = np.array(obs_dict[depth_name], copy=True)
        if depth.ndim == 3:  # (1, H, W) or (H, W, 1)
            depth = depth.squeeze()
        fig.add_subplot(rows, cols, plot_idx + len(CAMERAS))
        plt.imshow(depth, cmap="viridis")
        plt.axis('off')
        plt.title(f"{camera}_depth | step {ts}")

        plot_idx += 1

    plt.savefig("debug_plot.png")


#########################################################
##                    Main Program                     ##
#########################################################

if __name__ == "__main__":
    visualize_demo()
    exit(0)
