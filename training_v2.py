from seqeval.metrics import classification_report
from tqdm import tqdm
from utils import Config
import torchmetrics
import torch
import utils
import wandb
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_TO_IDS, IDS_TO_LABELS = utils.load_labels()
RELATIONS_TO_IDS, IDS_TO_RELATIONS = utils.load_relations()
CONFIG = utils.load_config(Config.CONFIG)


def valid(model, dataset, ner_accuracy, ner_f1_score, re_accuracy, re_f1_score, validation_loop=True):
    model.eval()

    eval_loss, eval_ner_accuracy, eval_ner_f1_score, eval_re_accuracy, eval_re_f1_score = 0, 0, 0, 0, 0
    eval_ner_preds, eval_ner_labels, eval_re_preds, eval_re_labels = [], [], [], []
    rows_count = len(dataset)

    with torch.inference_mode():
        for mask, ids, ner_labels, re_labels, object_position, subject_position in dataset:
            model_out = model(
                attention_mask=mask,
                input_ids=ids,
                ner_labels=ner_labels,
                re_labels=re_labels,
            )

            ner_targets = ner_labels.view(-1)
            ner_predictions = torch.argmax(model_out.ner_probs.view(-1, model.num_labels), axis=1)
            re_predictions = torch.argmax(model_out.re_probs, axis=1)

            active_labels = ner_labels.view(-1) != -100  # shape (batch_size, seq_len)

            ner_labels = torch.masked_select(ner_targets, active_labels)
            ner_predictions = torch.masked_select(ner_predictions, active_labels)

            eval_ner_labels.extend(ner_labels)
            eval_ner_preds.extend(ner_predictions)
            eval_re_labels.extend(re_labels)
            eval_re_preds.extend(re_predictions)

            total_loss = model_out.ner_loss + model_out.re_loss

            eval_loss += total_loss.item()
            eval_ner_accuracy += ner_accuracy(ner_predictions, ner_labels)
            eval_ner_f1_score += ner_f1_score(ner_predictions, ner_labels)
            eval_re_accuracy += re_accuracy(re_predictions, re_labels)
            eval_re_f1_score += re_f1_score(re_predictions, re_labels)

    if validation_loop:
        logg_value = {
            "Valid Loss": (eval_loss / rows_count),
            "Valid NER Accuracy": (eval_ner_accuracy / rows_count),
            "Valid NER F1 score": (eval_ner_f1_score / rows_count),
            "Valid RE Accuracy": (eval_re_accuracy / rows_count),
            "Valid RE F1 score": (eval_re_f1_score / rows_count),
        }
        wandb.log(logg_value, commit=True)
        return

    ner_labels = [IDS_TO_LABELS[id.item()] for id in eval_ner_labels]
    ner_predictions = [IDS_TO_LABELS[id.item()] for id in eval_ner_preds]
    re_labels = [IDS_TO_RELATIONS[id.item()] for id in eval_re_labels]
    re_predictions = [IDS_TO_RELATIONS[id.item()] for id in eval_re_preds]

    return ner_labels, ner_predictions, re_labels, re_predictions


@utils.wandb_log(CONFIG)
def fit(model, optimizer, scheduler, metrics, train_dataset, dev_dataset, test_dataset, device):
    ner_accuracy = torchmetrics.Accuracy('multiclass', num_classes=model.num_labels, average="weighted").to(device)
    ner_f1_score = torchmetrics.F1Score('multiclass', num_classes=model.num_labels, average="weighted").to(device)
    re_accuracy = torchmetrics.Accuracy('multiclass', num_classes=model.num_relations, average="weighted").to(device)
    re_f1_score = torchmetrics.F1Score('multiclass', num_classes=model.num_relations, average="weighted").to(device)

    def train_loop(epoch):
        tqdm_bar = tqdm(train_dataset)

        for index, (mask, ids, ner_labels, re_labels, object_position, subject_position) in enumerate(tqdm_bar):
            tqdm_bar.set_description_str(f"|| Epoch: {epoch:04} ||")

            model.train()
            optimizer.zero_grad()

            model_out = model(
                attention_mask=mask,
                input_ids=ids,
                ner_labels=ner_labels,
                re_labels=re_labels,
            )

            ner_targets = ner_labels.view(-1)
            active_labels = ner_targets != -100

            ner_predictions = torch.argmax(model_out.ner_probs.view(-1, model.num_labels), dim=1)
            re_predictions = torch.argmax(model_out.re_probs, dim=1)

            ner_labels = torch.masked_select(ner_targets, active_labels)
            ner_predictions = torch.masked_select(ner_predictions, active_labels)

            total_loss = model_out.ner_loss + model_out.re_loss
            total_loss.backward()

            metrics["loss"] += total_loss.item()
            metrics["ner_accuracy"] += ner_accuracy(ner_predictions, ner_labels)
            metrics["ner_f1_score"] += ner_f1_score(ner_predictions, ner_labels)
            metrics["re_accuracy"] += re_accuracy(re_predictions, re_labels)
            metrics["re_f1_score"] += re_f1_score(re_predictions, re_labels)

            if index % wandb.config["logging_interval"] == 0:
                current_index = metrics["index"] + index
                logg_value = {
                    "Train Loss": (metrics["loss"] / current_index),
                    "Train NER Accuracy": (metrics["ner_accuracy"] / current_index),
                    "Train NER F1 score": (metrics["ner_f1_score"] / current_index),
                    "Train RE Accuracy": (metrics["re_accuracy"] / current_index),
                    "Train RE F1 score": (metrics["re_f1_score"] / current_index),
                }
                wandb.log(
                    logg_value,
                    step=metrics["step"],
                    commit=False
                    if index % wandb.config["validation_interval"] == 0 else True,
                )
                metrics["step"] += (0 if index % wandb.config["validation_interval"] == 0 else 1)

            if index % wandb.config["validation_interval"] == 0:
                valid(model, dev_dataset, ner_accuracy, ner_f1_score, re_accuracy, re_f1_score)

                metrics["step"] += 1

            if wandb.config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=wandb.config["max_grad_norm"], norm_type=2.0)

            optimizer.step()
            scheduler.step()

        metrics["index"] += len(train_dataset)
        logg_value = {
            "Train Loss": (metrics["loss"] / current_index),
            "Train NER Accuracy": (metrics["ner_accuracy"] / current_index),
            "Train NER F1 score": (metrics["ner_f1_score"] / current_index),
            "Train RE Accuracy": (metrics["re_accuracy"] / current_index),
            "Train RE F1 score": (metrics["re_f1_score"] / current_index),
        }
        wandb.log(
            logg_value,
            step=metrics["step"],
            commit=False,
        )

        valid(model, dev_dataset, ner_accuracy, ner_f1_score, re_accuracy, re_f1_score)
        metrics["step"] += 1

    for epoch in range(wandb.config["epochs"]):
        train_loop(epoch)

    ner_labels, ner_predictions, re_labels, re_predictions = valid(model, test_dataset, ner_accuracy, ner_f1_score, re_accuracy, re_f1_score, validation_loop=False)  # noqa

    print(classification_report([ner_labels], [ner_predictions], zero_division=False))
    print(classification_report([[f'B-{item}' for item in re_labels]], [[f'B-{item}' for item in re_predictions]], zero_division=False))

    return model, optimizer, scheduler, metrics
