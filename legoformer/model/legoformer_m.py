from omegaconf import DictConfig

from legoformer import BackboneMultiView
from legoformer.model.legoformer_base import LegoFormer
from legoformer.util.utils import init_weights
import torch.nn as nn
import torch
from einops import rearrange


class LegoFormerM(LegoFormer):
    """
        Multi-View LegoFormer model definition
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        # Set up networks
        self.backbone = BackboneMultiView(n_out_channels=64)
        self.view_embedder = nn.Linear(64 * 8 * 8, self.d_model)

        # Initialize Backbone weights
        self.backbone.apply(init_weights)

    def images2tokens(self, images):
        """
            Maps input images into input tokens for the Transformer-Encoder.
        :param images: Input Images, shape: [B, N, 3, H (224), W (224)]
        :return: Input tokens
        """
        features = self.backbone(images)        # features.shape => [B, N, C (64), H (8), W (8)]
        features = rearrange(features, 'b n c h w -> b n (c h w)')
        tokens = self.view_embedder(features)   # tokens.shape => [B, N, D_MODEL]
        return tokens

    def get_decoder_mask(self):
        """
            Generate decoder-side attention mask
        :return: Attention mask
        """
        # Create boolean identity matrix
        tgt_mask = torch.ones((self.num_queries, self.num_queries), dtype=torch.bool, device=self.device)
        # Select diagonal entries
        tgt_mask_diag = torch.diagonal(tgt_mask)
        # Replace diagonal entries with False
        tgt_mask_diag[:] = False
        # Replace diagonals with -inf and everything else with 0
        tgt_mask = tgt_mask.float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
        return tgt_mask
