from typing import Any, Callable, Concatenate, Optional, ParamSpec, TypeVar, Dict
from functools import wraps
import re
import torch
import yaml
from typing import List
import os
import wandb
from enum import Enum

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
URI_PATTERN = r"(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*[^\W\ ]))"
SPLIT_PATTERN = r'([,.()"_])'
URL_LABEL = "URI-URI"

P = ParamSpec("P")
A = TypeVar("A")
B = TypeVar("B")


class Config(str, Enum):
    CONFIG = 'config'
    SWEEP_CONFIG = 'sweep_config'
    BACKUP = 'config_backup'

    def __str__(self):
        return self.value


def wandb_init(config_: Dict[str, str]):
    def wandb_init_(func: Callable[Concatenate[A, P], B]):
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[Any]:
            wandb.init(project=config_['project_name'] if config_ is not None else None, config=config_, dir=CURRENT_DIR)
            result = func(*args, **kwargs)
            wandb.finish()

            return result

        return wrapper

    return wandb_init_


def wandb_log(config_: Dict[str, str], save_model_state=True):
    def wandb_log_(func: Callable[Concatenate[A, P], B]):
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Optional[Any]:
            backup_config(wandb.config)
            wandb.run.log_code(f"{CURRENT_DIR}/", name="source", include_fn=lambda path: path.endswith((".py", ".ipynb", ".yaml")))

            model, optimizer, scheduler, metrics = func(*args, **kwargs)

            if save_model_state:
                model_path = f"{CURRENT_DIR}/models/{wandb.config['model']}.pt"

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "metrics": metrics,
                    },
                    model_path,
                )

                wandb_model = wandb.Artifact("model", type="model")
                wandb_model.add_file(model_path)
                wandb.log_artifact(wandb_model)

            return model, optimizer, scheduler, metrics

        return wrapper

    return wandb_log_


def load_model(model, project_name: str = 'dp-project-v2', model_name: str = 'model', model_version: str = 'latest',):
    artifact = wandb.run.use_artifact(f'kripso/{project_name}/{model_name}:{model_version}', type='model')
    artifact.download(f'{CURRENT_DIR}/models/')

    checkpoint = torch.load(f'{CURRENT_DIR}/models/{model_name}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def load_config(name: str):
    with open(f"{CURRENT_DIR}/conf/{name}.yaml") as f:
        data = yaml.safe_load(f)
    return data


def backup_config(config_):
    config_path = f"{CURRENT_DIR}/conf/{Config.BACKUP}.yaml"
    with open(config_path, 'w+') as f:
        yaml.safe_dump(dict(config_), f)

    artifact = wandb.Artifact("config", type="config")
    artifact.add_file(config_path)
    wandb.log_artifact(artifact)


def load_labels():
    """loads labels from config

    Returns:
        List[Dict[str, str]]: return two dictionaries -> labels_to_ids, ids_to_labels
    """
    with open(f"{CURRENT_DIR}/conf/labels.yaml") as f:
        data = yaml.safe_load(f)

    keys = []
    for key in data['entities'].values():
        if key != "O":
            keys.extend([f'B-{key}', f'I-{key}'])
        else:
            keys.append(key)

    labels_to_ids = {label: index for index, label in enumerate(keys)}
    ids_to_labels = {value: key for key, value in labels_to_ids.items()}

    return labels_to_ids, ids_to_labels


def load_relations():
    """loads labels from config

    Returns:
        List[Dict[str, str]]: return two dictionaries -> labels_to_ids, ids_to_labels
    """
    with open(f"{CURRENT_DIR}/conf/labels.yaml") as f:
        data = yaml.safe_load(f)

    relations_to_ids = {label: index for index, label in enumerate(data['relations'])}
    ids_to_relations = {value: key for key, value in relations_to_ids.items()}

    return relations_to_ids, ids_to_relations


def get_labels():
    _, ids_to_labels = load_labels()
    return [item for item in ids_to_labels.values()]


def get_relations():
    _, ids_to_relations = load_relations()
    return [item for item in ids_to_relations.values()]


def string_to_list_1(sentence: str) -> List[str]:
    urls = re.findall(URI_PATTERN, sentence)
    return [
        sub_item if sub_item != URL_LABEL else urls.pop(0)
        for sub_list in [re.split(SPLIT_PATTERN, item) for item in re.subn(URI_PATTERN, URL_LABEL, sentence)[0].split(" ")]
        for sub_item in sub_list
        if sub_item != ""
    ]


def string_to_list_2(sentence: str) -> List[str]:
    urls = re.findall(URI_PATTERN, sentence)
    clean_sentence = re.subn(URI_PATTERN, URL_LABEL, sentence)[0]

    items = []
    for item in clean_sentence.split(" "):
        for sub_item in re.split(SPLIT_PATTERN, item):
            if sub_item != "":
                items.append(sub_item if sub_item != URL_LABEL else urls.pop(0))
    return items


def custom_replace(string_: str, replace_mapping: List[str]) -> str:
    for old, new in replace_mapping:
        string_ = string_.replace(old, new)

    return string_


def list_to_string(list_: List[str]) -> str:
    replace_mapping = [
        (" _ ", "_"),
        (" .", "."),
        (" . ", ". "),
        (" , ", ", "),
        (" ( ", " ("),
        (" ) ", ") "),
    ]

    return custom_replace(" ".join(list_), replace_mapping)


def filter_ner(dataframe):
    return dataframe.drop_duplicates(subset=["tokens"], keep="first").loc[:, ["tokens", "labels"]].reset_index(drop=True)


def set_random_seeds(device: str, seed_value: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed_value (int, optional): Random seed to set. Defaults to 42.
        device (str, required): what device should torch use ['cuda','cpu']
    """
    torch.np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device == "cuda":
        torch.cuda.manual_seed(seed_value)
    else:
        torch.manual_seed(seed_value)


if __name__ == "__main__":
    print(load_labels())
