"""
2020/2/12:加入了 ALBERT
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from utils import calc_metric, find_triggers
from utils import report_to_telegram
import warnings

warnings.filterwarnings('ignore')
import numpy as np

import json
import os
from consts import NONE, PAD, CLS, SEP, UNK, TRIGGERS, ARGUMENTS, ENTITIES, POSTAGS
from utils import build_vocab

from migration_model.enet.corpus.Sentence import Sentence
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer, OpenAIGPTModel, OpenAIGPTTokenizer, CTRLModel, CTRLTokenizer, TransfoXLModel, TransfoXLTokenizer, XLNetModel, XLNetTokenizer, XLMModel, XLMTokenizer, DistilBertModel, DistilBertTokenizer, RobertaModel, RobertaTokenizer, XLMRobertaModel, XLMRobertaTokenizer, AlbertModel, AlbertTokenizer

# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS_dict = {
    'Bert': (BertModel, BertTokenizer, 'bert-base-uncased'),
    'Bert_large': (BertModel, BertTokenizer, 'bert-large-uncased'),
    "Gpt": (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
    "Gpt2": (GPT2Model, GPT2Tokenizer, 'gpt2'),
    "Ctrl": (CTRLModel, CTRLTokenizer, 'ctrl'),
    "TransfoXL": (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
    "Xlnet_base": (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
    "Xlnet_large": (XLNetModel, XLNetTokenizer, 'xlnet-large-cased'),
    "XLM": (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
    "DistilBert_base": (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
    "DistilBert_large": (DistilBertModel, DistilBertTokenizer, 'distilbert-large-cased'),
    "Roberta_base": (RobertaModel, RobertaTokenizer, 'roberta-base'),
    "Roberta_large": (RobertaModel, RobertaTokenizer, 'roberta-large'),
    "XLMRoberta_base": (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
    "XLMRoberta_large": (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-large'),
    "ALBERT-base-v1": (AlbertModel, AlbertTokenizer, 'albert-base-v1'),
    "ALBERT-large-v1": (AlbertModel, AlbertTokenizer, 'albert-large-v1'),
    "ALBERT-xlarge-v1": (AlbertModel, AlbertTokenizer, 'albert-xlarge-v1'),
    "ALBERT-xxlarge-v1": (AlbertModel, AlbertTokenizer, 'albert-xxlarge-v1'),
    "ALBERT-base-v2": (AlbertModel, AlbertTokenizer, 'albert-base-v2'),
    "ALBERT-large-v2": (AlbertModel, AlbertTokenizer, 'albert-large-v2'),
    "ALBERT-xlarge-v2": (AlbertModel, AlbertTokenizer, 'albert-xlarge-v2'),
    "ALBERT-xxlarge-v2": (AlbertModel, AlbertTokenizer, 'albert-xxlarge-v2'),
}

parser = argparse.ArgumentParser()
parser.add_argument("--PreTrainModel", type=str, default=str(list(MODELS_dict.keys())))
parser.add_argument("--early_stop", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--l2", type=float, default=1e-5)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--logdir", type=str, default="logdir")
parser.add_argument("--trainset", type=str, default="data/train.json")
parser.add_argument("--devset", type=str, default="data/dev.json")
parser.add_argument("--testset", type=str, default="data/test.json")
parser.add_argument("--LOSS_alpha", type=float, default=1.0)
parser.add_argument("--telegram_bot_token", type=str, default="")
parser.add_argument("--telegram_chat_id", type=str, default="")
parser.add_argument("--PreTrain_Model", type=str, default="Bert_large")
parser.add_argument("--test_path", type=str, default="data/one_word.json")

if os.name == "nt":
    parser.add_argument(
        "--model_path",
        type=str,
        default="Transformer-based-pretrained-model-for-event-extraction-master\save_model\\latest_model.pt")
    parser.add_argument("--batch_size", type=int, default=4)
else:
    parser.add_argument("--model_path", type=str, default="./train_log/ALBERT-base-v1/latest_model.pt")
    parser.add_argument("--batch_size", type=int, default=16)

hp = parser.parse_args()

if hp.PreTrain_Model not in MODELS_dict.keys():
    KeyError("PreTrain_Model不在可选列表内")

tokenizer = MODELS_dict[hp.PreTrain_Model][1].from_pretrained(MODELS_dict[hp.PreTrain_Model][2])

# init vocab
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)


class ACE2005Dataset(data.Dataset):

    def __init__(self, fpath):
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.arguments_li, self.adjm_li = [], [], [], [], [], []

        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item['words']
                entities = [[NONE] for _ in range(len(words))]
                triggers = [NONE] * len(words)
                postags = item['pos-tags']
                sentence = Sentence(json_content=item)
                adjm = (sentence.adjpos, sentence.adjv)
                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }

                for entity_mention in item['golden-entity-mentions']:
                    arguments['candidates'].append(
                        (entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))

                    for i in range(entity_mention['start'], entity_mention['end']):
                        entity_type = entity_mention['entity-type']
                        if i == entity_mention['start']:
                            entity_type = 'B-{}'.format(entity_type)
                        else:
                            entity_type = 'I-{}'.format(entity_type)

                        if len(entities[i]) == 1 and entities[i][0] == NONE:
                            entities[i][0] = entity_type
                        else:
                            entities[i].append(entity_type)

                for event_mention in item['golden-event-mentions']:
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        trigger_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)

                    event_key = (event_mention['trigger']['start'], event_mention['trigger']['end'],
                                 event_mention['event_type'])
                    arguments['events'][event_key] = []
                    for argument in event_mention['arguments']:
                        role = argument['role']
                        if role.startswith('Time'):
                            role = role.split('-')[0]
                        arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))
                self.sent_li.append([CLS] + words + [SEP])
                self.entities_li.append([[PAD]] + entities + [[PAD]])
                self.postags_li.append([PAD] + postags + [PAD])
                self.triggers_li.append(triggers)
                self.arguments_li.append(arguments)
                self.adjm_li.append(adjm)

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, postags, triggers, arguments, adjm = self.sent_li[idx], self.entities_li[idx], self.postags_li[
            idx], self.triggers_li[idx], self.arguments_li[idx], self.adjm_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads = [], [], [], []
        for w, e, p in zip(words, entities, postags):
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [
                w
            ]  ## w=offenses,而tokens= ['offense', '##s'],此时只保留offense,否则会导致触发词的漂移量错位
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            p = [p] + [PAD] * (len(tokens) - 1)
            e = [e] + [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision
            p = [postag2idx[postag] for postag in p]
            e = [[entity2idx[entity] for entity in entities] for entities in e]

            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers, adjm

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


def pad(batch):
    tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm = list(
        map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        postags_x_2d[i] = postags_x_2d[i] + [0] * (maxlen - len(postags_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))]

    return tokens_x_2d, entities_x_3d, postags_x_2d, \
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, triggers_2d, adjm


def eval(model, iterator, fname):
    model.eval()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm = batch

            trigger_loss, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.module.predict_triggers(
                tokens_x_2d=tokens_x_2d,
                entities_x_3d=entities_x_3d,
                postags_x_2d=postags_x_2d,
                head_indexes_2d=head_indexes_2d,
                triggers_y_2d=triggers_y_2d,
                arguments_2d=arguments_2d,
                adjm=adjm)

            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(
                    argument_hidden, argument_keys, arguments_2d, adjm)
                arguments_hat_all.extend(argument_hat_2d)
                # if i == 0:

                #     print("=====sanity check for triggers======")
                #     print('triggers_y_2d[0]:', triggers_y_2d[0])
                #     print("trigger_hat_2d[0]:", trigger_hat_2d[0])
                #     print("=======================")

                #     print("=====sanity check for arguments======")
                #     print('arguments_y_2d[0]:', arguments_y_2d[0])
                #     print('argument_hat_1d[0]:', argument_hat_1d[0])
                #     print("arguments_2d[0]:", arguments_2d)
                #     print("argument_hat_2d[0]:", argument_hat_2d)
                #     print("=======================")
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

    arg_write = arguments_all[0]['events']
    for arg_key in arg_write:
        arg = arg_write[arg_key]  # list,eg: [(0, 5, 25), (8, 19, 17), (20, 21, 29)]
        for ii, tup in enumerate(arg):
            arg[ii] = (tup[0], tup[1], idx2argument[tup[2]])  # 将id 转为 str
        arg_write[arg_key] = arg

    arghat_write = arguments_hat_all[0]['events']
    for arg_key in arghat_write:
        arg = arghat_write[arg_key]  # list,eg: [(0, 5, 25), (8, 19, 17), (20, 21, 29)]
        for ii, tup in enumerate(arg):
            arg[ii] = (tup[0], tup[1], idx2argument[tup[2]])  # 将id 转为 str
        arghat_write[arg_key] = arg

    print('#真实值#\t{}\n'.format(arg_write))
    print('#预测值#\t{}\n'.format(arghat_write))
    print("\n")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 20 + " 超参 " + "=" * 20)
    for para in hp.__dict__:
        print(" " * (20 - len(para)), para, "=", hp.__dict__[para])
    PreModel = MODELS_dict[hp.PreTrain_Model][0].from_pretrained(MODELS_dict[hp.PreTrain_Model][2])

    if os.path.exists(hp.model_path):
        print('=======载入模型=======')
        model = torch.load(hp.model_path)

    test_dataset = ACE2005Dataset(hp.test_path)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    if not os.path.exists(os.path.split(hp.model_path)[0]):
        os.makedirs(os.path.split(hp.model_path)[0])

    eval(model, test_iter, os.path.join(hp.logdir, '0') + '_test')