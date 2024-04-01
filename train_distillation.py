import hydra
import os
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(version_base = "1.3", config_path = "config", config_name = "main_config")
def main(config: DictConfig) -> None:
    if config.device  == 'cuda':
        if not torch.cuda.is_available():
            logging.info('GPU not available: switching to CPU')
            config.device = 'cpu'

    setup_seed(config.seed)

    shutil.move(config.teacher_model_path,'teacher_model.pth')


    ds_parser = #hydra.utils.instantiate dataset_parser
    trainer   = Trainer(config, ds_parser)

    trainer.train()