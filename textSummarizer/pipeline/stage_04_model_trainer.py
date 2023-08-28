from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
import optuna
from textSummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)

        def objective(trial):
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
            num_train_epochs = trial.suggest_int("num_train_epochs", 5, 30)
            batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)

            model_trainer.train(learning_rate, num_train_epochs, batch_size, weight_decay)
            return model_trainer.get_validation_loss()

        study = optuna.create_study(direction="minimize")
        n_trials = 100
        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        logger.info("Best trial:")
        logger.info("Value:", best_trial.value)
        logger.info("Params:")

        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
