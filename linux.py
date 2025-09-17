##############################################################################################
## linux.py                                                                                 ##
## This versions is an implementation of training the PerAct model using the full           ##
## dependencies on linux instead of the lightweight version in main.py.                     ##
## ---------------------------------------------------------------------------------------- ##
## Author:   Mathias Otnes                                                                  ##
## Date:     09/15/2025                                                                     ##
##############################################################################################

#########################################################
##                    Environment                      ##
#########################################################

import os
import sys
import numpy as np
import torch
import clip

sys.path.insert(0, os.path.dirname(os.path.join(os.path.dirname(__file__), "src", "yarr")))
sys.path.insert(0, os.path.dirname(os.path.join(os.path.dirname(__file__), "src", "rlbench")))
sys.path.insert(0, os.path.dirname(os.path.join(os.path.dirname(__file__), "src", "arm")))

np.bool = np.bool_ # bad trick to fix numpy version issue :(

from src.utils.utils import create_replay, fill_replay
from src.yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer

os.environ["DISPLAY"] = ":0"
os.environ["PYOPENGL_PLATFORM"] = "egl"

#########################################################
##                   Configuration                     ##
#########################################################

# constants
TASK                        = 'open_drawer'
DATA_FOLDER                 = 'src/data'
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
VOXEL_SIZES     = [20]  # 20x20x20 voxels
NUM_LATENTS     = 512   # PerceiverIO latents
SCENE_BOUNDS    = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6] # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
BATCH_SIZE      = 1
NUM_DEMOS       = 8     # total number of training demonstrations to use while training PerAct
NUM_TEST        = 2     # episodes to evaluate on
REPLAY_SIZE     = 1e5   # max size of replay buffer


#########################################################
##                      Storage                        ##
#########################################################

# Storing replay buffer to disc instead of RAM
sys.path.append('src')
data_path = os.path.join(DATA_FOLDER, EPISODES_FOLDER)

train_replay_storage_dir = 'replay_train'
if not os.path.exists(train_replay_storage_dir):
  os.mkdir(train_replay_storage_dir)

test_replay_storage_dir = 'replay_test'
if not os.path.exists(test_replay_storage_dir):
  os.mkdir(test_replay_storage_dir)

#########################################################
##                    Main Program                     ##
#########################################################

if __name__ == "__main__":
    
    
    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE,
        timesteps=1,
        save_dir=train_replay_storage_dir,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        replay_size=REPLAY_SIZE
    )

    test_replay_buffer = create_replay(
        batch_size=BATCH_SIZE,
        timesteps=1,
        save_dir=test_replay_storage_dir,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        replay_size=REPLAY_SIZE
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("RN50", device=device) # CLIP-ResNet50
    
    
    print("-- Train Buffer --")
    fill_replay(
        replay=train_replay_buffer,
        start_idx=0,
        num_demos=NUM_DEMOS,
        demo_augmentation=True,
        demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
        cameras=CAMERAS,
        rlbench_scene_bounds=SCENE_BOUNDS,
        voxel_sizes=VOXEL_SIZES,
        rotation_resolution=ROTATION_RESOLUTION,
        crop_augmentation=False,
        clip_model=clip_model,
        device=device
    )

    print("-- Test Buffer --")
    fill_replay(
        replay=test_replay_buffer,
        start_idx=NUM_DEMOS,
        num_demos=NUM_TEST,
        demo_augmentation=True,
        demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
        cameras=CAMERAS,
        rlbench_scene_bounds=SCENE_BOUNDS,
        voxel_sizes=VOXEL_SIZES,
        rotation_resolution=ROTATION_RESOLUTION,
        crop_augmentation=False,
        clip_model=clip_model,
        device=device
    )

    # delete the CLIP model since we have already extracted language features
    del clip_model

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(train_replay_buffer)
    train_dataset = train_wrapped_replay.dataset()
    train_data_iter = iter(train_dataset)

    test_wrapped_replay = PyTorchReplayBuffer(test_replay_buffer)
    test_dataset = test_wrapped_replay.dataset()
    test_data_iter = iter(test_dataset)