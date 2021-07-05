import pytorch_lightning as pl
import torch
import torch.nn as nn


class OutputLayer(pl.LightningModule):
    """
        Output Layer of the LegoFormer
        Maps Transformer's output vector to decomposition factors.
    """

    def __init__(self, dim_model, output_resolution=32):
        """

        :param dim_model:           Transformer model output dimensionality
        :param output_resolution:   Output voxel grid resolution (side length)
        """
        super().__init__()

        # Define output layers
        self.linear_z = nn.Linear(dim_model, output_resolution)
        self.linear_y = nn.Linear(dim_model, output_resolution)
        self.linear_x = nn.Linear(dim_model, output_resolution)

        # Initialize output layers
        torch.nn.init.xavier_uniform_(self.linear_z.weight)
        torch.nn.init.xavier_uniform_(self.linear_y.weight)
        torch.nn.init.xavier_uniform_(self.linear_x.weight)

    def forward(self, x):
        """

        :param x: input with shape [BATCH_SIZE, NUM_FACTORS, DIM_MODEL]
        :return: 3 vectors (decomposition factors), each with shape [BATCH_SIZE, NUM_FACTORS, NUM_VOXEL (32)]
        """
        z_factors = self.linear_z(x).sigmoid()
        y_factors = self.linear_y(x).sigmoid()
        x_factors = self.linear_x(x).sigmoid()

        return z_factors, y_factors, x_factors
