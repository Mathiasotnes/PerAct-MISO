##############################################################################################
## qfunctions.py                                                                            ##
## Implements PerAct-MISO specific versions of qfunctions based on:                         ##
## https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing    ##
## ---------------------------------------------------------------------------------------- ##
## Author:   Mathias Otnes                                                                  ##
## Date:     09/17/2025                                                                     ##
##############################################################################################

#########################################################
##                 PerAct-MISO imports                 ##
#########################################################

from src.utils.voxelgrid import VoxelGrid


#########################################################
##                      Libraries                      ##
#########################################################

import copy
import torch
import torch.nn as nn


#########################################################
##                     Public API                      ##
#########################################################

class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 voxel_grid: VoxelGrid,
                 rotation_resolution: float,
                 device,
                 training):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxel_grid = voxel_grid
        self._qnet = copy.deepcopy(perceiver_encoder)
        self._qnet._dev = device

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self,
                obs,
                proprio,
                pcd,
                lang_goal_embs,
                bounds=None):

        # flatten point cloud
        bs = obs[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)

        # flatten rgb
        image_features = [o[0] for o in obs]
        feat_size = image_features[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in
             image_features], 1)

        # voxelize
        voxel_grid = self._voxel_grid.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # batch bounds if necessary
        if bounds.shape[0] != bs:
            bounds = bounds.repeat(bs, 1)

        # forward pass
        q_trans, rot_and_grip_q, collision_q = self._qnet(voxel_grid,
                                                          proprio,
                                                          lang_goal_embs,
                                                          bounds)
        return q_trans, rot_and_grip_q, collision_q, voxel_grid

    def latents(self):
        return self._qnet.latent_dict