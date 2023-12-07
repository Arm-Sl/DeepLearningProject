import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
import torch.utils.data as data_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if config["model"]["load"]:
        print(f"======= Loading {config['model_params']['name']} =======")
        model = vae_models[config['model_params']['name']](**config['model_params'])
        model.load_state_dict(torch.load(os.path.join(config["model"]["path"], "state_dict_model.pt")))
        model.eval()
        experiment = VAEXperiment(model,config['exp_params'])
        data = VAEDataset(**config["data_params"])
        data.setup()
        """
        0: 3
        1: 2
        2: 1
        3: 18
        4: 4
        5: 8
        6: 11
        7: 0
        8: 61
        9: 7
        """
        d = [data.val_dataset_concat[3][0],
             data.val_dataset_concat[2][0],
             data.val_dataset_concat[1][0],
             data.val_dataset_concat[18][0],
             data.val_dataset_concat[4][0],
             data.val_dataset_concat[8][0],
             data.val_dataset_concat[11][0],
             data.val_dataset_concat[0][0],
             data.val_dataset_concat[61][0],
             data.val_dataset_concat[7][0],
            ]
        #experiment.visualize_latent_space(data.test_dataloader())
        #experiment.visualize_effect(config['model_params']["latent_dim"])
        experiment.visualize_each_dim(d, config['model_params']["latent_dim"])
    else:
        tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                    name=config['model_params']['name'],)

        # For reproducibility
        seed_everything(config['exp_params']['manual_seed'], True)

        model = vae_models[config['model_params']['name']](**config['model_params'])
        experiment = VAEXperiment(model,
                                config['exp_params'])

        data = VAEDataset(**config["data_params"])#, pin_memory=len(config['trainer_params']['gpus']) != 0)

        data.setup()
        runner = Trainer(logger=tb_logger,
                        callbacks=[
                            LearningRateMonitor(),
                            ModelCheckpoint(save_top_k=2, 
                                            dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                            monitor= "val_loss",
                                            save_last= True),
                        ],
                        #strategy=DDPStrategy(find_unused_parameters=False),
                        **config['trainer_params'])


        Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
        Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
        Path(f"{tb_logger.log_dir}/Model").mkdir(exist_ok=True, parents=True)


        print(f"======= Training {config['model_params']['name']} =======")
        runner.fit(experiment, datamodule=data)
    