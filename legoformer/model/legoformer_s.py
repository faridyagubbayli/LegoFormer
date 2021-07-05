import itertools

import torch
from einops import rearrange
from omegaconf import DictConfig

from legoformer import BackboneSingleView
from legoformer.model.legoformer_base import LegoFormer
from legoformer.util.utils import init_weights, positionalencoding2d


class LegoFormerS(LegoFormer):
    """
        Single-View LegoFormer model definition
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        # Set up the backbone
        self.backbone = BackboneSingleView(n_out_channels=48)
        # Initialize Backbone weights
        self.backbone.apply(init_weights)

    def images2tokens(self, images):
        # images.shape => [B, N, 3, H (224), W (224)]
        features = self.backbone(images)

        patches = self.split_features(features)
        patches = self.add_2d_pos_enc(patches)
        tokens = rearrange(patches, 'b n np d -> b (n np) d')  # tokens.shape => [B, num_patches (49), d_model (768))
        return tokens

    def get_decoder_mask(self):
        return None

    def split_features(self, features):
        """
            Splits image features into patches
        :features: Image features
        :return: Image patches
        """
        features = features.split(4, -1)
        features = [im.split(4, -2) for im in features]

        features = list(itertools.chain.from_iterable(features))
        features = [m.flatten(2, -1) for m in features]
        features = torch.stack(features, 2)
        return features

    def add_2d_pos_enc(self, patches):
        """
            Adds 2D sine-cos positional encoding to the image patches
        :param patches:
        :return:
        """
        pos_enc_2d = positionalencoding2d(768, 8, 8)[:, :7, :7]
        pos_enc_2d = rearrange(pos_enc_2d, 'd h w -> h w d')
        pos_enc_2d = rearrange(pos_enc_2d, 'h w d -> 1 1 (h w) d').to(self.device)  # [1 1 49 256]
        return patches + pos_enc_2d
