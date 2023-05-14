import yaml
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(name: str):
    with open(f"{os.path.join(CURRENT_DIR, os.pardir)}/conf/{name}.yaml") as f:
        data = yaml.safe_load(f)
    return data


def normalize_glove(tokens):
    mapping = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '{',
        '-RCB-': '}',
    }
    data = []
    for token in tokens:
        if token in mapping:
            token = mapping[token]
        data.append(token)
    return data

SHORTEN = load_config('labels')


def process_labels(labels):

    data = []
    for label in labels:
        data.append(f'B-{SHORTEN.get(label,"O")}' if SHORTEN.get(label,"O") != 'O' else 'O')
        
    data = [
        data[0],
        *[(data[index].replace('B-', 'I-') if (data[index].startswith('B-') and data[index] == data[index-1]) else data[index]) for index in range(1, len(data))]
    ]
    return data