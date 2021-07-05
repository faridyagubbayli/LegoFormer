"""
    ShapeNet dataset definition

    Resources used to develop this script:
        - https://github.com/hzxie/Pix2Vox/blob/master/utils/data_loaders.py
"""
from typing import Tuple, List

import cv2
import json
import numpy as np
import os
import random
import scipy.io
import scipy.ndimage
import torch.utils.data.dataset

from datetime import datetime as dt

from omegaconf import DictConfig

import legoformer.util.binvox_rw as binvox_rw


class ShapeNetDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, cfg: DictConfig, dataset_type, transforms, opts):
        self.dataset_type = dataset_type
        self.transforms = transforms

        self.n_views            = opts['n_views']
        self.repeat_factor      = opts['repeat_factor']
        self.n_vox              = opts['n_vox']
        self.selection_mode     = opts['selection_mode']

        self.check_options()

        self.file_list          = self.build_file_list(cfg, dataset_type)
        self.real_size          = len(self.file_list)

    def check_options(self) -> None:
        """
            Validate given arguments
        :return:
        """
        assert self.dataset_type in ['train', 'val', 'test']
        assert isinstance(self.repeat_factor, int) and self.repeat_factor > 0
        assert isinstance(self.n_views, int) and 1 <= self.n_views <= 24
        assert isinstance(self.n_vox, int) and self.n_vox > 0
        assert self.selection_mode in ['random', 'fixed']

    def __len__(self) -> int:
        """
            Return the dataset size.

            It is possible to repeat the training dataset by `repeat_factor` times.
            This helps to have longer epochs
                and avoid losing time when initializing the workers at the epoch start.
        :return: Dataset size
        """
        if self.dataset_type == 'train':
            return self.repeat_factor * self.real_size
        else:
            return self.real_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """
            Get a single dataset item
        :param idx: Item position in the file_list
        :return: A set of images, GT volume and sample info
        """
        idx = idx % self.real_size
        taxonomy_name, sample_name, images, volume = self.get_datum(idx)

        if self.transforms:
            # Apply pre-processing to the images
            images = self.transforms(images)

        return images, volume, (taxonomy_name, sample_name)

    def get_datum(self, idx: int) -> Tuple[str, str, np.ndarray, np.ndarray]:
        """
            Load a single data sample
        :param idx: Position of the data sample in the file list
        :return: Sample info, images and GT volume
        """
        # Grab the sample info and paths
        sample                  = self.file_list[idx]
        taxonomy_name           = sample['taxonomy_name']       # string, like '02691156'
        sample_name             = sample['sample_name']         # string, like '6c432109eee42aed3b053f623496d8f5'
        all_image_paths         = sample['image_paths']
        volume_path             = sample['volume']

        # Select images to use
        selected_image_paths = self.select_images(self.selection_mode, all_image_paths, self.n_views)

        # Read RGB Images
        images = [self.read_img(image_path) for image_path in selected_image_paths]
        images = np.asarray(images)  # images.shape => [N_VIEWS, H (137), W(137), C (4)]

        # Read 3D GT Volume
        volume = self.read_volume(volume_path)  # volume.shape => [N_VOX (32), N_VOX (32), N_VOX (32)]

        return taxonomy_name, sample_name, images, volume

    def build_file_list(self, cfg: DictConfig, dataset_type):
        """
            Builds a list of all files
        :param cfg: Dataset configuration
        :param dataset_type: Type of the dataset [train | val | test]
        :return: List of all files
        """
        # Grab image and volume paths
        taxonomy_path                   = cfg.taxonomy_path
        image_path_template             = cfg.image_path
        volume_path_template            = cfg.voxel_path

        # Load all taxonomies of the dataset
        with open(taxonomy_path, encoding='utf-8') as file:
            dataset_taxonomy = json.loads(file.read())

        files = []

        # Load data for each category
        for taxonomy in dataset_taxonomy:
            # Command-line update
            self.log_taxonomy_loading(taxonomy, dataset_type)

            # Get a list of files for a single category
            taxonomy_id     = taxonomy['taxonomy_id']
            samples         = taxonomy[dataset_type]
            taxonomy_files  = self.get_files_of_taxonomy(taxonomy_id, samples,
                                                         image_path_template, volume_path_template)

            # Combine category-level file list to the overall list
            files.extend(taxonomy_files)

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return files

    @staticmethod
    def get_files_of_taxonomy(taxonomy_folder_id, samples, image_path_template, volume_path_template):
        """
            Builds the file list of a single category
        :param taxonomy_folder_id: Category ID
        :param samples: List of sample IDs
        :param image_path_template: Template path to the images folder
        :param volume_path_template: Template path to the volume folder
        :return: List of files of a single category
        """
        TOTAL_VIEWS                 = 24
        all_image_indices           = range(TOTAL_VIEWS)
        files_of_taxonomy           = []

        for sample_idx, sample_name in enumerate(samples):
            # List of all images for a single sample
            image_paths = [
                image_path_template % (taxonomy_folder_id, sample_name, image_idx)
                for image_idx in all_image_indices
            ]

            # Path to the ground-truth volume
            volume_file_path = volume_path_template % (taxonomy_folder_id, sample_name)

            # Append to the list of files
            files_of_taxonomy.append({
                'taxonomy_name':    taxonomy_folder_id,
                'sample_name':      sample_name,
                'image_paths':      image_paths,
                'volume':           volume_file_path
            })
        return files_of_taxonomy

    @staticmethod
    def select_images(selection_mode: str, all_image_paths: List[str], num_images: int) -> List[str]:
        """
            Select images to use for a single sample
        :param selection_mode: Whether to use first N images or randomly choose them
        :param all_image_paths: Path to all images
        :param num_images: Number of images to choose
        :return: List of selected image paths
        """
        if selection_mode == 'random':
            selected_ids = random.sample(range(len(all_image_paths)), num_images)
            selected_image_paths = [all_image_paths[i] for i in selected_ids]
        else:
            selected_image_paths = [all_image_paths[i] for i in range(num_images)]
        return selected_image_paths

    @staticmethod
    def read_img(path: str) -> np.ndarray:
        """
            Read an image at the given path
        :param path: Path to the image
        :return: Loaded image, shape: [H (137), W (137), C (4)]
        """
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # Ensure that the image is either RGB or RGBA
        assert len(image.shape) in [3, 4],  \
            print('[FATAL] %s It seems that there is something wrong with the image file %s' % (dt.now(), path))
        return image

    @staticmethod
    def read_volume(path):
        """
            Read 3D volume at the given path
        :param path: Path to the volume
        :return: Loaded 3D volume, shape: [N_VOX (32), N_VOX (32), N_VOX (32)]
        """
        _, extension = os.path.splitext(path)

        if extension == '.mat':
            volume = scipy.io.loadmat(path)
            volume = volume['Volume'].astype(np.float32)
        elif extension == '.binvox':
            with open(path, 'rb') as f:
                volume = binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)
        return volume

    @staticmethod
    def log_taxonomy_loading(taxonomy, dataset_type):
        """
            Update command-line with taxonomy loading message
        :param taxonomy: Taxonomy info
        :param dataset_type: Type of the dataset [train | val | test]
        :return:
        """
        sample_count = len(taxonomy[dataset_type])
        print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s] %d' %
              (dt.now(), taxonomy['taxonomy_id'], taxonomy['taxonomy_name'], sample_count))
