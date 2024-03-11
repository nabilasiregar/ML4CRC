# to run -->$ python3 Hydra_Python.py --multirun
# to modify parameters, see optuna_config.yaml file in 'conf' directory.
# Hydra manages configuration and Optuna does hyperparameter optimization.
# Optuna calls train_model various times with the different configurations that you define in optuna_config.yaml.
# Each trial will be registered by Hydra.

import hydra
from omegaconf import DictConfig
import optuna
import type_your_model_here  # Library with YOUR ML model


@hydra.main(config_path="conf", config_name="config") # replace conf and config with their actual path
def train_model(cfg: DictConfig) -> float:
    # Accessing Optuna configurations. BTW you can always change them in optuna_config.yaml
    # Scroll down and look for 'Search Space'
    learning_rate = cfg.learning_rate
    batch_size = cfg.batch_size
    num_layers = cfg.num_layers

    # Model training logic
    model = type_your_model_here.YourModel(learning_rate, batch_size, num_layers)
    val_loss = model.train()

    # Return the validation loss as the objective to MINIMIZE or change it to MAXIMIZE in optuna_config.yaml
    # look for 'direction'
    return val_loss


if __name__ == "__main__":
    train_model()