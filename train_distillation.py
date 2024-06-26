import hydra
import os
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil

from src.utils import *
from src.training.Trainer import *



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(version_base = "1.3", config_path = "config", config_name = "train_distillation_config")
def main(config: DictConfig) -> None:
    if config.device  == 'cuda':
        if not torch.cuda.is_available():
            logging.info('GPU not available: switching to CPU')
            config.device = 'cpu'

    setup_seed(config.seed)

    config['log_location'] = HydraConfig.get().runtime.output_dir
    config.model.n_channels = config['n_channels']

    shutil.copy(config.teacher_model_path,f'{config["log_location"]}/teacher_model.pth')


    ds_parser = hydra.utils.instantiate(config.dataset)
    trainer   = Trainer(config, ds_parser)

    trainer.train()
    trainer.test()



if __name__=="__main__":
    main()