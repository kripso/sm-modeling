from seqeval.metrics import classification_report
from tqdm import tqdm
from utils import Config
import torchmetrics
import torch
import utils
import wandb
import os
import logging


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_TO_IDS, IDS_TO_LABELS = utils.load_labels()
CONFIG = utils.load_config(Config.CONFIG)


def valid(model, dataset, accuracy, f1_score, validation_loop=True):
    model.eval()

    eval_loss, eval_accuracy, eval_f1_score = 0, 0, 0
    eval_preds, eval_labels = [], []

    with torch.inference_mode():
        for mask, ids, labels in dataset:
            loss, preds = model(attention_mask=mask, input_ids=ids, labels=labels)

            flattened_targets = labels.view(-1)
            flattened_predictions = torch.argmax(
                preds.view(-1, model.num_labels), axis=1
            )

            active_labels = labels.view(-1) != -100  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_labels)
            predictions = torch.masked_select(flattened_predictions, active_labels)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            eval_loss += loss.item()
            eval_accuracy += accuracy(predictions, labels)
            eval_f1_score += f1_score(predictions, labels)

    if validation_loop:
        logg_value = {
            "Valid Loss": (eval_loss / len(dataset)),
            "Valid Accuracy": (eval_accuracy / len(dataset)),
            "Valid F1 score": (eval_f1_score / len(dataset)),
        }
        if wandb.run is not None:
            wandb.log(logg_value, commit=True)
        else:
            logging.info(logg_value)
        return

    labels = [IDS_TO_LABELS[id.item()] for id in eval_labels]
    predictions = [IDS_TO_LABELS[id.item()] for id in eval_preds]

    return labels, predictions


@utils.wandb_log(CONFIG)
def fit(model, optimizer, scheduler, metrics, train_dataset, dev_dataset, test_dataset, device):
    accuracy = torchmetrics.Accuracy(num_classes=model.num_labels, average="weighted").to(device)
    f1_score = torchmetrics.F1Score(num_classes=model.num_labels, average="weighted").to(device)

    def train_loop(epoch):
        tqdm_bar = tqdm(train_dataset)

        for index, (mask, ids, labels) in enumerate(tqdm_bar):
            tqdm_bar.set_description_str(f"|| Epoch: {epoch:04} ||")

            model.train()
            optimizer.zero_grad()

            loss, preds = model(attention_mask=mask, input_ids=ids, labels=labels)

            flattened_targets = labels.view(-1)
            flattened_predictions = torch.argmax(preds.view(-1, model.num_labels), axis=1)

            active_labels = labels.view(-1) != -100

            labels = torch.masked_select(flattened_targets, active_labels)
            predictions = torch.masked_select(flattened_predictions, active_labels)

            metrics["loss"] += loss.item()
            metrics["accuracy"] += accuracy(predictions, labels)
            metrics["f1_score"] += f1_score(predictions, labels)

            if index % wandb.config["logging_interval"] == 0:
                current_index = metrics["index"] + index
                logg_value = {
                    "Train Loss": (metrics["loss"] / current_index),
                    "Train Accuracy": (metrics["accuracy"] / current_index),
                    "Train F1 score": (metrics["f1_score"] / current_index),
                }
                if wandb.run is not None:
                    wandb.log(
                        logg_value,
                        step=metrics["step"],
                        commit=False
                        if index % wandb.config["validation_interval"] == 0
                        else True,
                    )
                else:
                    logging.info(logg_value)
                metrics["step"] += (
                    0 if index % wandb.config["validation_interval"] == 0 else 1
                )

            if index % wandb.config["validation_interval"] == 0:
                valid(model, dev_dataset, accuracy, f1_score)

                metrics["step"] += 1

            loss.backward()

            if wandb.config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=wandb.config["max_grad_norm"], norm_type=2.0)

            optimizer.step()
            scheduler.step()

        metrics["index"] += len(train_dataset)
        logg_value = {
            "Train Loss": (metrics["loss"] / metrics["index"]),
            "Train Accuracy": (metrics["accuracy"] / metrics["index"]),
            "Train F1 score": (metrics["f1_score"] / metrics["index"]),
        }
        if wandb.run is not None:
            wandb.log(
                logg_value,
                step=metrics["step"],
                commit=False,
            )
        else:
            logging.info(logg_value)

        valid(model, dev_dataset, accuracy, f1_score)
        metrics["step"] += 1

    for epoch in range(wandb.config["epochs"]):
        train_loop(epoch)

    labels, predictions = valid(model, test_dataset, accuracy, f1_score, validation_loop=False)

    print(classification_report([labels], [predictions], zero_division=False))
    return model, optimizer, scheduler, metrics
