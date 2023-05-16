import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel, BertPreTrainedModel, BertConfig, BertTokenizerFast
from typing import List
import utils
from utils import Config
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Tuple, Any
from dataclasses import fields
import random

LABELS_TO_IDS, IDS_TO_LABELS = utils.load_labels()
RELATIONS_TO_IDS, IDS_TO_RELATIONS = utils.load_relations()
CONFIG = utils.load_config(Config.CONFIG)


@dataclass
class BaseModelOutput(OrderedDict):
    def __post_init__(self):
        for field in fields(self):
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __iter__(self):
        return iter(self.to_tuple())

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] for k in self.keys())


@dataclass
class JointNERAndREModelOutput(BaseModelOutput):
    ner_loss: Optional[torch.Tensor] = None
    ner_probs: Optional[torch.Tensor] = None
    re_loss: Optional[torch.Tensor] = None
    re_probs: Optional[torch.Tensor] = None


class CustomDataset(Dataset):
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["bert_model_string"])

    def __init__(self, dataframe, device):
        self.data = dataframe.reindex()
        self.len = len(dataframe)
        self.device = device

    @staticmethod
    def tokenize(tokens: List[str], is_split=True, return_tensors=None):
        return CustomDataset.tokenizer(
            tokens if is_split else utils.string_to_list_1(tokens),
            truncation=True,
            padding="max_length",
            max_length=CONFIG["max_sentence_lenght"],
            add_special_tokens=True,
            return_offsets_mapping=True,
            is_split_into_words=True,
            return_tensors=return_tensors
            # return_token_type_ids=True,
        )

    def as_tensor(self, item, as_type=torch.long):
        return torch.as_tensor(item).to(self.device, dtype=as_type)

    def __getitem__(self, index, as_type=torch.long):
        tokens = self.data["tokens"][index]
        labels = self.data["labels"][index]

        encoded = CustomDataset.tokenize(tokens)

        encoded_labels = np.ones(CONFIG["max_sentence_lenght"], dtype=int) * -100

        i = -1
        for index, mapping in enumerate(encoded["offset_mapping"]):
            if mapping[1] != 0:
                if mapping[0] == 0:
                    i += 1
                encoded_labels[index] = LABELS_TO_IDS.get(labels[i], 0)

        return {
            "attention_mask": self.as_tensor(encoded["attention_mask"]),
            "input_ids": self.as_tensor(encoded["input_ids"]),
            "labels": self.as_tensor(encoded_labels),
        }

    def __len__(self):
        return self.len


class JointNERAndREDataset(CustomDataset):
    def __init__(self, dataframe, device, train=False):
        super().__init__(dataframe, device)
        self.data_relations = self.data[self.data['relation'] != 'no_relation'].reset_index(drop=True)
        self.train = train

    def __getitem__(self, index, as_dict=False, as_type=torch.long):
        data = self.data.iloc[index]

        if self.train and self.data["relation"][index] == 'no_relation' and random.random() <= 0.5:
            idx = int(random.randint(0, len(self.data_relations) - 1))
            data = self.data_relations.iloc[idx]

        tokens = data["tokens"]
        labels = data["labels"]

        # Encode tokens
        encoded = self.tokenize(tokens)

        # Convert NER labels to token-level labels
        encoded_labels = np.ones(CONFIG["max_sentence_lenght"], dtype=int) * -100
        encoded_relation = RELATIONS_TO_IDS.get(data["relation"])
        i = -1
        for index, mapping in enumerate(encoded["offset_mapping"]):
            if mapping[1] != 0:
                if mapping[0] == 0:
                    i += 1
                encoded_labels[index] = LABELS_TO_IDS.get(labels[i], 0)

        result = OrderedDict(
            attention_mask=self.as_tensor(encoded["attention_mask"]),
            input_ids=self.as_tensor(encoded["input_ids"]),
            labels=self.as_tensor(encoded_labels),
            relations=self.as_tensor(encoded_relation),
            object_position=self.as_tensor(data["objectSpan"]),
            subject_position=self.as_tensor(data["subjectSpan"]),
        )

        return dict(result) if as_dict else list(result.values())


class JointNERAndREModel(nn.Module):
    config = BertConfig.from_pretrained(CONFIG["bert_model_string"])

    def __init__(self, labels: List[str] = None, relations: List[str] = None, re_class_weights=None):
        super(JointNERAndREModel, self).__init__()
        labels = labels if labels is not None else utils.load_labels()
        relations = relations if relations is not None else utils.load_relations()

        self.num_relations = len(relations)
        self.num_labels = len(labels)
        self.relations = relations
        self.labels = labels
        self.re_class_weights = re_class_weights

        self.bert_model = BertModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.ner_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.num_labels * 2),
            nn.Linear(self.num_labels * 2, self.num_labels),
        )

        self.re_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.num_relations * 2),
            nn.Tanh(),
            nn.Linear(self.num_relations * 2, self.num_relations),
        )

        self.init_weights()

    def init_weights(self):
        BertPreTrainedModel._init_weights(self, self.ner_classifier)
        BertPreTrainedModel._init_weights(self, self.re_classifier)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, ner_labels=None, re_labels=None):
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(bert_output.last_hidden_state)

        ner_logits = self.ner_classifier(sequence_output)
        ner_probs = torch.softmax(ner_logits, dim=2)

        # re_cls_ner_in = torch.cat((sequence_output[:, 0, :], ner_logits.max(dim=2).values), dim=1)
        re_cls_ner_in = torch.cat((sequence_output[:, 0, :], bert_output.pooler_output), dim=1)
        re_logits = self.re_classifier(re_cls_ner_in)
        re_probs = torch.softmax(re_logits, dim=1)

        ner_loss = None
        if ner_labels is not None:
            ner_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ner_loss = ner_loss_fct(ner_logits.view(-1, self.num_labels), ner_labels.view(-1))

        re_loss = None
        if re_labels is not None:
            re_loss_fct = nn.CrossEntropyLoss(weight=self.re_class_weights)
            re_loss = re_loss_fct(re_logits, re_labels)

        return JointNERAndREModelOutput(
            ner_loss=ner_loss,
            ner_probs=ner_probs,
            re_loss=re_loss,
            re_probs=re_probs,
        )
