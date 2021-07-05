import torch.utils.data.dataset
from omegaconf import DictConfig
import pytorch_lightning as pl
import legoformer.data as transforms
from legoformer.data import ShapeNetDataset


class ShapeNetDataModule(pl.LightningDataModule):
    """
        A PyTorch-Lightning DataModule subclass for the ShapeNet
    """
    DATASET_NAME = 'ShapeNet'

    def __init__(self, cfg_data: DictConfig):
        super().__init__()
        self.cfg_data           = cfg_data

        cfg_const               = cfg_data.constants
        self.IMG_SIZE           = cfg_const.img_h, cfg_const.img_w
        self.CROP_SIZE          = cfg_const.crop_img_h, cfg_const.crop_img_w
        self.n_views            = cfg_const.n_views
        self.n_vox              = cfg_const.n_vox
        self.selection_mode     = cfg_const.selection_mode
        self.bg_mode            = cfg_const.bg_mode
        self.train_augmentation = cfg_const.train_augmentation

        # Hold dataset options separately
        self.dataset_opts = {
            'n_views': self.n_views,
            'n_vox': self.n_vox,
            'repeat_factor': self.cfg_data.loader.repeat_factor,
            'selection_mode': self.selection_mode
        }

    def setup(self, stage=None) -> None:
        """
            Called from PyTorch-Lightning framework
        :param stage:
        :return:
        """
        pass  # Nothing to do

    def train_dataloader(self):
        cfg_dt              = self.cfg_data.transforms
        cfg_dl              = self.cfg_data.loader
        cfg_ds              = self.cfg_data.dataset[self.DATASET_NAME]

        # Set up data augmentation
        train_transforms    = self.get_train_transforms(cfg_dt)
        # Initialize dataset
        train_dataset       = ShapeNetDataset(cfg_ds, 'train', train_transforms, self.dataset_opts)

        # Initialize dataloader
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg_dl.batch_size,
            num_workers=cfg_dl.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True)

        return train_data_loader

    def val_dataloader(self):
        cfg_dt               = self.cfg_data.transforms
        cfg_dl               = self.cfg_data.loader
        cfg_ds               = self.cfg_data.dataset[self.DATASET_NAME]

        # Set up data augmentation
        val_transforms = self.get_eval_transforms(cfg_dt)
        # Initialize dataset
        val_dataset = ShapeNetDataset(cfg_ds, 'val', val_transforms, self.dataset_opts)

        # Initialize dataloader
        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=cfg_dl.batch_size,
            num_workers=cfg_dl.num_workers,
            pin_memory=True,
            shuffle=True)

        return val_data_loader

    def test_dataloader(self):
        cfg_dt              = self.cfg_data.transforms
        cfg_ds              = self.cfg_data.dataset[self.DATASET_NAME]

        # Set up data augmentation
        test_transforms = self.get_eval_transforms(cfg_dt)
        # Initialize dataset
        test_dataset = ShapeNetDataset(cfg_ds, 'test', test_transforms, self.dataset_opts)

        # Initialize dataloader
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False)

        return test_data_loader

    def get_test_taxonomy_file_path(self):
        ds_name = self.cfg_data.loader.test_dataset
        return self.cfg_data.dataset[ds_name].taxonomy_path

    def get_train_transforms(self, cfg_dt: DictConfig) -> transforms.Compose:
        """
            Build training time pre-processing pipeline
        :param cfg_dt: Data Transform Config
        :return: Composed pre-processing transforms
        """
        if self.bg_mode == 'random':
            bg_transform = transforms.RandomBackground(cfg_dt.train_rand_bg_color_range)
        else:
            bg_transform = transforms.FixedBackground()

        if self.train_augmentation:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(self.IMG_SIZE, self.CROP_SIZE),
                bg_transform,
                transforms.ColorJitter(cfg_dt.brightness, cfg_dt.contrast, cfg_dt.saturation),
                transforms.RandomNoise(cfg_dt.noise_std),
                transforms.Normalize(cfg_dt.mean, std=cfg_dt.std),
                transforms.RandomFlip(),
                transforms.RandomPermuteRGB(),
                transforms.ChangeOrdering(),
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(self.IMG_SIZE, self.CROP_SIZE),
                bg_transform,
                transforms.Normalize(cfg_dt.mean, std=cfg_dt.std),
                transforms.ChangeOrdering(),
            ])
        return train_transforms

    def get_eval_transforms(self, cfg_dt: DictConfig) -> transforms.Compose:
        """
            Build evaluation time pre-processing pipeline
        :param cfg_dt: Data Transform Config
        :return: Composed pre-processing transforms
        """
        return transforms.Compose([
            transforms.CenterCrop(self.IMG_SIZE, self.CROP_SIZE),
            transforms.RandomBackground(cfg_dt.test_rand_bg_color_range),
            transforms.Normalize(mean=cfg_dt.mean, std=cfg_dt.std),
            transforms.ChangeOrdering()
        ])
