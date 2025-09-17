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
import time

from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.join(os.path.dirname(__file__), "src", "yarr")))
sys.path.insert(0, os.path.dirname(os.path.join(os.path.dirname(__file__), "src", "rlbench")))
sys.path.insert(0, os.path.dirname(os.path.join(os.path.dirname(__file__), "src", "arm")))

np.bool = np.bool_ # bad trick to fix numpy version issue :(

from src.utils.replay import create_replay, fill_replay
from src.utils.voxelgrid import VoxelGrid
from src.utils.perceiverio import PerceiverIO
from src.utils.agent import PerceiverActorAgent
from src.yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from src.arm.utils import stack_on_channel, discrete_euler_to_quaternion, get_gripper_render_pose

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
##                  Helper Functions                   ##
#########################################################

def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0

def _preprocess_inputs(replay_sample):
    obs, pcds = [], []
    for n in CAMERAS:
        rgb = stack_on_channel(replay_sample['%s_rgb' % n])
        pcd = stack_on_channel(replay_sample['%s_point_cloud' % n])

        rgb = _norm_rgb(rgb)

        obs.append([rgb, pcd]) # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd) # only pointcloud
    return obs, pcds


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
        image_size=IMAGE_SIZE,
        low_dim_size=LOW_DIM_SIZE,
        replay_size=REPLAY_SIZE
    )

    test_replay_buffer = create_replay(
        batch_size=BATCH_SIZE,
        timesteps=1,
        save_dir=test_replay_storage_dir,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        image_size=IMAGE_SIZE,
        low_dim_size=LOW_DIM_SIZE,
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
        data_path = data_path,
        variation_descriptions_pkl = VARIATION_DESCRIPTIONS_PKL,
        episode_folder = EPISODE_FOLDER,
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
        data_path = data_path,
        variation_descriptions_pkl = VARIATION_DESCRIPTIONS_PKL,
        episode_folder = EPISODE_FOLDER,
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
    
    # initialize voxelizer
    vox_grid = VoxelGrid(
        coord_bounds=SCENE_BOUNDS,
        voxel_size=VOXEL_SIZES[0],
        device=device,
        batch_size=BATCH_SIZE,
        feature_size=3,
        max_num_coords=np.prod([IMAGE_SIZE, IMAGE_SIZE]) * len(CAMERAS),
    )

    # sample from dataset
    batch = next(train_data_iter)
    lang_goal = batch['lang_goal'][0][0][0]
    batch = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}

    # preprocess observations
    obs, pcds = _preprocess_inputs(batch)

    # flatten observations
    bs = obs[0][0].shape[0]
    pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcds], 1)

    image_features = [o[0] for o in obs]
    feat_size = image_features[0].shape[1]
    flat_imag_features = torch.cat(
        [p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in image_features], 1)

    # tensorize scene bounds
    bounds = torch.tensor(SCENE_BOUNDS, device=device).unsqueeze(0)

    # voxelize!
    voxel_grid = vox_grid.coords_to_bounding_voxel_grid(pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)

    # swap to channels fist
    vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()

    # expert action voxel indicies
    vis_gt_coord = batch['trans_action_indicies'][:, -1, :3].int().detach().cpu().numpy()
    
    # initialize PerceiverIO
    perceiver_encoder = PerceiverIO(
        depth=6,
        iterations=1,
        voxel_size=VOXEL_SIZES[0],
        initial_dim=3 + 3 + 1 + 3,
        low_dim_size=4,
        layer=0,
        num_rotation_classes=72,
        num_grip_classes=2,
        num_collision_classes=2,
        num_latents=NUM_LATENTS,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        activation='lrelu',
        input_dropout=0.1,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        voxel_patch_size=5,
        voxel_patch_stride=5,
        final_dim=64,
    )
    
    # initialize PerceiverActor
    peract_agent = PerceiverActorAgent(
        coordinate_bounds=SCENE_BOUNDS,
        perceiver_encoder=perceiver_encoder,
        camera_names=CAMERAS,
        batch_size=BATCH_SIZE,
        voxel_size=VOXEL_SIZES[0],
        voxel_feature_size=3,
        num_rotation_classes=72,
        rotation_resolution=5,
        lr=0.0001,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        lambda_weight_l2=0.000001,
        transform_augmentation=False,
        optimizer_type='lamb',
    )
    peract_agent.build(training=True, cameras=CAMERAS, device=device, image_size=IMAGE_SIZE)
    
    LOG_FREQ = 1
    TRAINING_ITERATIONS = 5

    start_time = time.time()
    for iteration in range(TRAINING_ITERATIONS):
        batch = next(train_data_iter)
        batch = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}
        update_dict = peract_agent.update(iteration, batch)

        if iteration % LOG_FREQ == 0:
            elapsed_time = (time.time() - start_time) / 60.0
            print("Total Loss: %f | Elapsed Time: %f mins" % (update_dict['total_loss'], elapsed_time))
            
    batch = next(test_data_iter)
    lang_goal = batch['lang_goal'][0][0][0]
    batch = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}
    update_dict = peract_agent.update(iteration, batch, backprop=False)

    # things to visualize
    vis_voxel_grid = update_dict['voxel_grid'][0].detach().cpu().numpy()
    vis_trans_q = update_dict['q_trans'][0].detach().cpu().numpy()
    vis_trans_coord = update_dict['pred_action']['trans'][0].detach().cpu().numpy()
    vis_gt_coord = update_dict['expert_action']['action_trans'][0].detach().cpu().numpy()

    # discrete to continuous
    continuous_trans = update_dict['pred_action']['continuous_trans'][0].detach().cpu().numpy()
    continuous_quat = discrete_euler_to_quaternion(update_dict['pred_action']['rot_and_grip'][0][:3].detach().cpu().numpy(),
                                                resolution=peract_agent._rotation_resolution)
    gripper_open = bool(update_dict['pred_action']['rot_and_grip'][0][-1].detach().cpu().numpy())
    ignore_collision = bool(update_dict['pred_action']['collision'][0][0].detach().cpu().numpy())

    # gripper visualization pose
    voxel_size = 0.045
    voxel_scale = voxel_size * 100
    gripper_pose_mat = get_gripper_render_pose(voxel_scale, SCENE_BOUNDS[:3], continuous_trans, continuous_quat)
    
    