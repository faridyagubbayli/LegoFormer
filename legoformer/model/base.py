import pytorch_lightning as pl
from legoformer.util.metrics import *
import torch.nn.functional as F
import torch.optim as optim
from legoformer.util.utils import positionalencoding1d
from einops import repeat


class BaseModel(pl.LightningModule):
    """
        Base class for all models.
        Contains helper methods to set up the experiments and the training.
    """

    def __init__(self):
        """
            The initialization is left to the subclasses.
        """
        super().__init__()
        pass

    def configure_optimizers(self):
        """
            Method that will be called by the PyTorch-Lightning framework.
            The optimizer (AdaGrad) and linear warm-up learning rate scheduler is set up here.
        :return: optimizer and LR scheduler
        """

        warmup_steps = self.cfg_optim.warmup_steps

        def lambda_optim(epoch):
            lr_coeff = 1
            if self.global_step < warmup_steps:
                # training is in the warm-up phase, adjust learning rate linearly
                lr_coeff = (self.global_step + 1) / warmup_steps
            # make sure that LR doesn't go beyond 1
            return min(1, lr_coeff)

        base_lr = self.cfg_optim.lr
        # Setup optimizer
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, self.parameters()), lr=base_lr)
        # Setup scheduler
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_optim)
        return (
            [optimizer],
            [
                {
                    # Warm-up scheduler should adjust the LR after each optimization step
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            ]
        )

    def prepare_learned_queries(self, learned_queries: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
            Prepare the learned decomposition factor queries for the decoder
            Adds sine-cos positional encoding and replicates single-set of queries among the batch dimension
        :param learned_queries: Single set of learned queries, shape: [num_queries (12), d_model (768)]
        :param batch_size: Size of the batch
        :return: Learned decomposition factor queries
        """

        # Add positional encoding
        # pos_enc.shape => same as `learned_queries` | [num_queries (12), d_model (768)]
        pos_enc = positionalencoding1d(self.d_model, self.num_queries).to(self.device)
        learned_queries = learned_queries + pos_enc

        # Expand (replicate) among the batch dimension
        learned_queries = repeat(learned_queries, 'nq dmodel -> nq b dmodel', b=batch_size)
        # learned_queries.shape => [num_queries (12), B_SIZE, d_model (768)]

        return learned_queries

    @staticmethod
    def binarize_preds(predictions: torch.Tensor, threshold=0.5) -> torch.Tensor:
        """
            Apply threshold on the predictions
        :param predictions: Predicted voxel grid
        :param threshold: Threshold limit
        :return: Binarized voxel grid
        """
        return predictions.__ge__(threshold).int()

    @staticmethod
    def calculate_loss(pred_vol: torch.Tensor, gt_vol: torch.Tensor) -> torch.Tensor:
        """
            Calculate the loss between predicted and ground-truth voxel grids
        :param pred_vol: Predicted voxel grid, size: [B, 32, 32, 32], dtype: float [0, 1]
        :param gt_vol: Ground-truth voxel grid, size: [B, 32, 32, 32], dtype: int {0, 1}
        :return: Calculated loss, scalar, dtype: float
        """
        # Behaviour of the MSE-loss changes when shapes don't match. Therefore, make sure shapes are the same.
        assert pred_vol.shape == gt_vol.shape
        return F.mse_loss(pred_vol, gt_vol)

    def log_metrics(self, pred_volume: torch.Tensor, gt_volume: torch.Tensor, tag: str) -> dict:
        """
            Log metrics during training and evaluation
        :param pred_volume: Predicted voxel grid, size: [B, 32, 32, 32], dtype: float [0, 1]
        :param gt_volume: Ground-truth voxel grid, size: [B, 32, 32, 32], dtype: int {0, 1}
        :param tag: Phase, {'train', 'val', 'test'}
        :return: Calculated metrics
        """
        assert tag in ['train', 'val', 'test']

        # Binarize the outputs
        pred_volume = self.binarize_preds(pred_volume, threshold=0.3)
        # Get the metrics requested by the user to be logged
        metrics = self.cfg_optim.metrics

        iou                 = calculate_iou(pred_volume, gt_volume)
        dice                = calculate_dice(pred_volume, gt_volume)
        occupation_ratio    = calculate_occupation_ratio(pred_volume, gt_volume)

        # Calculate F1-Score.
        # Calculating this metric is expensive. Therefore, avoid calculating it if the user did not request.
        f1 = calculate_fscore(pred_volume, gt_volume) if 'f1' in metrics else None

        available_metrics = {
            'iou': iou,
            'dice': dice,
            'occupation_ratio': occupation_ratio,
            'f1': f1
        }

        # Log only the metrics request by the user
        for metric in metrics:
            self.log_scalar(f'metrics/{tag}_{metric}', available_metrics[metric])
        return available_metrics

    def log_scalar(self, name, value) -> None:
        """
            Helper method to log the scalar values
        :param name: Metric name
        :param value: Metric value
        :return: None
        """
        # Log IoU metrics to the progress bar
        log_to_prog_bar = 'iou' in name
        self.log(name, value, prog_bar=True)

    def get_progress_bar_dict(self):
        """
            Method that will be called by the PyTorch-Lightning framework.
            Modify the progress-bar logs.
        :return:
        """
        # don't show the loss as it's None
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
