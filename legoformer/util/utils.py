import os

import numpy as np
import open3d
import mcubes
import math
import torch
from omegaconf import OmegaConf, DictConfig


def voxel_grid_to_mesh(vox_grid: np.array) -> open3d.geometry.TriangleMesh:
    """
        taken from: https://github.com/lmb-freiburg/what3d

        Converts a voxel grid represented as a numpy array into a mesh.
    """
    sp = vox_grid.shape
    if len(sp) != 3 or sp[0] != sp[1] or \
            sp[1] != sp[2] or sp[0] == 0:
        raise ValueError("Only non-empty cubic 3D grids are supported.")
    padded_grid = np.pad(vox_grid, ((1, 1), (1, 1), (1, 1)), 'constant')
    m_vert, m_tri = mcubes.marching_cubes(padded_grid, 0)
    m_vert = m_vert / (padded_grid.shape[0] - 1)
    out_mesh = open3d.geometry.TriangleMesh()
    out_mesh.vertices = open3d.utility.Vector3dVector(m_vert)
    out_mesh.triangles = open3d.utility.Vector3iVector(m_tri)
    return out_mesh


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def init_weights(m):
    """
        Utility method to initialize the backbone network weights

        Source: https://github.com/hzxie/Pix2Vox/blob/master/utils/network_utils.py
    :param m:
    :return:
    """
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_config(cfg: DictConfig, dir: str) -> None:
    """
        Save configuration to a file
    :param cfg: Configuration dict or DictConfig
    :param dir: Path to the save folder
    :return: None
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, 'config_model.yaml')
    OmegaConf.save(cfg, path)


def load_config(cfg_path: str) -> DictConfig:
    """
        Load configuration file. `base_config.yaml` is taken as a base for every config.
    :param cfg_path: Path to the configuration file
    :return: Loaded configuration
    """
    base_cfg = OmegaConf.load('legoformer/config/base_config.yaml')
    curr_cfg = OmegaConf.load(cfg_path)
    return OmegaConf.merge(base_cfg, curr_cfg)
