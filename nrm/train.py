import argparse
from typing import Type

import torch
import optuna
from tqdm import tqdm

from nrm.logger import Logger
from nrm.model import Model, Torus, OccupancyNetwork, MLP, Shell
from nrm.dataset.loader import TrainingSet, ValidationSet


def main(model_class: Type[Model],
         hyperparameter: dict,
         epochs: int,
         batch_size: int,
         early_stopping: int,
         lr: float,
         trial: optuna.Trial = None):
    device = torch.device("cuda")

    training_set = TrainingSet(batch_size, True)
    validation_set = ValidationSet(batch_size, False)

    model = model_class(**hyperparameter).to(device)
    #model = torch.compile(model)
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logger = Logger(device, model_class, hyperparameter, epochs, batch_size, early_stopping, lr, trial, training_set,
                    validation_set, model, loss_function)

    min_loss = torch.inf
    early_stopping_counter = 0

    for e in range(epochs):
        model.train()
        for batch_idx, (morph, pose, label) in enumerate(tqdm(training_set, desc=f"Training")):
            morph = morph.to(device, non_blocking=True)
            pose = pose.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            model.zero_grad()

            logit = model(morph, pose)
            loss = loss_function(logit, label.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            logger.log_training(e, batch_idx, morph, pose, label, logit, loss)

        model.eval()
        loss = 0.0
        for batch_idx, (morph, pose, label) in enumerate(tqdm(validation_set, desc=f"Validation")):
            morph = morph.to(device, non_blocking=True)
            pose = pose.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logit = model.predict(morph, pose)
            loss += loss_function(logit, label.float())

            logger.log_validation(e, batch_idx, morph, pose, label, logit, loss)
        loss /= len(validation_set) * batch_size

        if loss < min_loss:
            min_loss = loss
            early_stopping_counter = 0
            logger.save_model()
        else:
            early_stopping_counter += 1
            if early_stopping_counter == early_stopping:
                (print('Early Stopping'))
                return min_loss
        if trial is not None:
            trial.report(loss, e)
            if trial.should_prune():
                del logger
                raise optuna.TrialPruned()

    return min_loss


if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()
    args.model_class = eval(args.model_class)

    main(**vars(args), hyperparameter={"encoder_config": {"dim_encoding": 128},
                                       "decoder_config": {}})
