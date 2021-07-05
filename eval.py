#
# Developed by Farid Yagubbayli <faridyagubbayli@gmail.com> | <farid.yagubbayli@tum.de>
#

import argparse
import pytorch_lightning as pl
from legoformer.data import ShapeNetDataModule
from legoformer.util.utils import load_config
from legoformer import LegoFormerM, LegoFormerS


model_zoo = {
    'legoformer_m': LegoFormerM,
    'legoformer_s': LegoFormerS,
}


if __name__ == '__main__':
    # Get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config file", type=str)
    parser.add_argument("ckpt_path", help="Model checkpoint path", type=str)
    parser.add_argument("views", help="Number of views", type=int)
    args = parser.parse_args()

    config_path = args.config_path
    ckpt_path = args.ckpt_path
    n_views = args.views

    # Load config file
    cfg = load_config(config_path)

    # Enforce some config parameters
    cfg.trainer.precision = 32
    cfg.data.constants.n_views = n_views
    cfg.optimization.metrics = ['iou', 'f1']

    if cfg.seed != -1:
        pl.seed_everything(cfg.seed)

    net_type = cfg.network.type
    print('Network type:', net_type, ' n_views:', n_views)

    # Load model and data module
    model = model_zoo[cfg.network.type]
    model = model.load_from_checkpoint(ckpt_path, config=cfg)
    data_module = ShapeNetDataModule(cfg.data)

    # Start evaluation process
    trainer = pl.Trainer(callbacks=None, logger=False, **cfg.trainer)
    trainer.test(model, test_dataloaders=data_module.test_dataloader())
