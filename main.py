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
import clip
import torch
from typing import List

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
    
colab_yarr_path = os.path.join(os.path.dirname(__file__), "peract_colab", "yarr")
if os.path.isdir(colab_yarr_path):
    sys.path.insert(0, os.path.dirname(colab_yarr_path))
    print("[Info] Using yarr from peract_colab (dev mode).")
else:
    print("[Info] Using yarr from mod/Yarr (full install).")
    
colab_arm_path = os.path.join(os.path.dirname(__file__), "peract_colab", "arm")
if os.path.isdir(colab_arm_path):
    sys.path.insert(0, os.path.dirname(colab_arm_path))
    print("[Info] Using arm from peract_colab (dev mode).")
else:
    print("[Info] Using arm from mod/arm (full install).")

from rlbench.backend.observation import Observation

from rlbench.utils import get_stored_demo
from rlbench.backend.utils import extract_obs
from rlbench.demo import Demo

from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer

import arm.utils as utils

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
##                    Replay Buffer                    ##
#########################################################

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py

def create_replay(
    batch_size: int,
    timesteps: int,
    save_dir: str,
    cameras: list,
    voxel_sizes,
    replay_size=3e5
    ) -> ReplayBuffer:
    """ Creates a replay buffer to store demonstrations and agent experience.

    Args:
        batch_size (int): Batch size
        timesteps (int): Timesteps
        save_dir (str): Directory to save replay buffer
        cameras (list): List of camera names
        voxel_sizes (_type_): List of voxel sizes
        replay_size (_type_, optional): Replay size. Defaults to 3e5.

    Returns:
        ReplayBuffer: Replay buffer object
    """

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = [ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32)]

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(ObservationElement('%s_rgb' % cname, (3, IMAGE_SIZE, IMAGE_SIZE,), np.float32))
        observation_elements.append(ObservationElement('%s_depth' % cname, (1, IMAGE_SIZE, IMAGE_SIZE,), np.float32))
        observation_elements.append(ObservationElement('%s_point_cloud' % cname, (3, IMAGE_SIZE, IMAGE_SIZE,), np.float32)) # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,), np.int32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,), np.int32),
        ReplayElement('ignore_collisions', (ignore_collisions_size,), np.int32),
        ReplayElement('gripper_pose', (gripper_pose_size,), np.float32),
        ReplayElement('lang_goal_embs', (max_token_seq_len, lang_emb_dim,), np.float32), # extracted from CLIP's language encoder
        ReplayElement('lang_goal', (1,), object), # language goal string for debugging and visualization
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
    ]

    replay_buffer = UniformReplayBuffer( # all tuples in the buffer have equal sample weighting
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,), # 3 translation + 4 rotation quaternion + 1 gripper open
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer

# From https://github.com/stepjam/ARM/blob/main/arm/demo_loading_utils.py

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def _keypoint_discovery(demo: Demo,
                        stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                        last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def visualize_keypoint() -> None:
    """ Visualizes keypoints from a demo. """
    episode_idx_to_visualize = 1
    demo = get_stored_demo(data_path=data_path, index=episode_idx_to_visualize)

    # total timesteps
    print("Demo %s | %s total steps" % (episode_idx_to_visualize, len(demo._observations)))

    # use the heuristic to extract keyframes (aka keypoints)
    episode_keypoints = _keypoint_discovery(demo)

    # visualize rgb observations from these keyframes
    for kp_idx, kp in enumerate(episode_keypoints):
        obs_dict = extract_obs(demo._observations[kp], CAMERAS, t=kp)

        fig = plt.figure(figsize=(5, 5))
        rgb_name = "front_rgb"
        rgb = np.transpose(obs_dict[rgb_name], (1, 2, 0))
        plt.imshow(rgb)
        plt.axis('off')
        plt.title("front_rgb | step %s | keypoint %s " % (kp, kp_idx))
        plt.savefig(f"debug_plot_{kp_idx}.png")
        
# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
        obs_tp1: Observation,
        obs_tm1: Observation,
        rlbench_scene_bounds: List[float], # metric 3D bounds of the scene
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
        [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates

# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

# add individual data points to replay
def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool,
        description: str = '',
        clip_model = None,
        device = 'cpu') -> None:
    
    prev_action = None
    obs = inital_obs
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]
        trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
            obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes,
            rotation_resolution, crop_augmentation)

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(obs, CAMERAS, t=k, prev_action=prev_action)
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict['lang_goal_embs'] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        others = {'demo': True}
        final_obs = {
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
            'gripper_pose': obs_tp1.gripper_pose,
            'lang_goal': np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1

    # final step
    obs_dict_tp1 = extract_obs(obs_tp1, CAMERAS, t=k + 1, prev_action=prev_action)
    obs_dict_tp1['lang_goal_embs'] = lang_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)

def fill_replay(replay: ReplayBuffer,
        start_idx: int,
        num_demos: int,
        demo_augmentation: bool,
        demo_augmentation_every_n: int,
        cameras: List[str],
        rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool,
        clip_model = None,
        device = 'cpu'):
    print('Filling replay ...')
    for d_idx in range(start_idx, start_idx+num_demos):
        print("Filling demo %d" % d_idx)
        demo = get_stored_demo(data_path=data_path,
                               index=d_idx)

        # get language goal from disk
        varation_descs_pkl_file = os.path.join(data_path, EPISODE_FOLDER % d_idx, VARIATION_DESCRIPTIONS_PKL)
        with open(varation_descs_pkl_file, 'rb') as f:
          descs = pickle.load(f)

        # extract keypoints
        episode_keypoints = _keypoint_discovery(demo)

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0: # choose only every n-th frame
                continue

            obs = demo[i]
            desc = descs[0]
            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            _add_keypoints_to_replay(
                replay, obs, demo, episode_keypoints, cameras,
                rlbench_scene_bounds, voxel_sizes,
                rotation_resolution, crop_augmentation, description=desc,
                clip_model=clip_model, device=device)
    print('Replay filled with demos.')
    

#########################################################
##                    Train PerAct                     ##
#########################################################



#########################################################
##                    Main Program                     ##
#########################################################

if __name__ == "__main__":
    
    # visualize_demo()
    
    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE,
        timesteps=1,
        save_dir=train_replay_storage_dir,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES
    )

    test_replay_buffer = create_replay(
        batch_size=BATCH_SIZE,
        timesteps=1,
        save_dir=test_replay_storage_dir,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES
    )
    
    # visualize_keypoint()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("RN50", device=device) # CLIP-ResNet50
    print("Using device:", device)
    

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
    
    exit(0)
