"""
    Multi-view and Single-view backbone classes.

    Backbone is used to embed the high-dimensional RGB images into the low-dimensional representations.

    Resources used to develop this script:
        - https://github.com/hzxie/Pix2Vox/blob/master/models/encoder.py
"""

import torch
import torchvision.models
import pytorch_lightning as pl


class BackboneBase(pl.LightningModule):

    def __init__(self):
        super().__init__()

        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16.features.children()))[:18]
        # Freeze VGG16 model
        for param in vgg16.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        """
            Map input images to features
        :param images: Input Images, shape: [B, N, 3, H (224), W (224)]
        :return: Features, shape: [B, N, 64, 8, 8]
        """
        raise NotImplementedError('Should be implemented in the subclass!')

    def followup_conv(self, features: torch.Tensor) -> torch.Tensor:
        """
            Apply convolution blocks to the VGG-16 output
        :param features: VGG-16 output features, shape: [B, 512, 28, 28]
        :return: Lower dimensional features
        """
        raise NotImplementedError('Should be implemented in the subclass!')


class BackboneMultiView(BackboneBase):
    def __init__(self, n_out_channels: int = 64):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, n_out_channels, kernel_size=1),
            torch.nn.BatchNorm2d(n_out_channels),
            torch.nn.ReLU()
        )

    def forward(self, images):
        """
            Map input images to features
        :param images: Input Images, shape: [B, N, 3, H (224), W (224)]
        :return: Features, shape: [B, N, 64, 8, 8]
        """
        images = images.permute(1, 0, 2, 3, 4).contiguous()
        images = torch.split(images, 1, dim=0)
        all_img_features = []

        for img in images:
            single_img_features = self.vgg(img.squeeze(dim=0))              # shape => [B, 512, 28, 28]
            single_img_features = self.followup_conv(single_img_features)   # shape => [B, 64,   8,  8]
            all_img_features.append(single_img_features)

        all_img_features = torch.stack(all_img_features)
        all_img_features = all_img_features.permute(1, 0, 2, 3, 4).contiguous()  # shape => [B, N, 64, 8, 8]
        return all_img_features

    def followup_conv(self, features: torch.Tensor) -> torch.Tensor:
        """
            Apply convolution blocks to the VGG-16 output
        :param features: VGG-16 output features, shape: [B, 512, 28, 28]
        :return: Lower dimensional features
        """
        features = self.layer1(features)        # shape => [B, 512, 26, 26]
        features = self.layer2(features)        # shape => [B, 512,  8,  8]
        features = self.layer3(features)        # shape => [B, 64,   8,  8]
        return features


class BackboneSingleView(BackboneBase):
    def __init__(self, n_out_channels: int = 64):
        super().__init__()

        self.channel_reducer = torch.nn.Sequential(
            torch.nn.Conv2d(512, n_out_channels, kernel_size=1),
            torch.nn.BatchNorm2d(n_out_channels),
            torch.nn.ReLU()
        )
        torch.nn.init.xavier_uniform_(self.channel_reducer[0].weight)

    def followup_conv(self, features: torch.Tensor) -> torch.Tensor:
        """
            Apply convolution blocks to the VGG-16 output
        :param features: VGG-16 output features
        :return: Lower dimensional features
        """
        return self.channel_reducer(features)

    def forward(self, images):
        """
            Map input images to features
        :param images: Input Images, shape: [B, 1, 3, H (224), W (224)]
        :return: Features
        """
        n_views = images.shape[1]
        assert n_views == 1

        img_features = self.vgg(images[:, 0])
        img_features = self.followup_conv(img_features)
        img_features = img_features.unsqueeze(dim=1)
        return img_features
