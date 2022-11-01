import json
from multiprocessing import context
import token
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import tqdm

from sklearn import metrics

import os

class RaceExample(object):
    """ RACE 数据集的样本格式
    Args:
        race_id: data id
        context_sentence: article
        question: question
        ending_0/1/2/3: option_0/1/2/3
        label: true answer
    """


    def __init__(self,
                 race_id,
                 context_sentence,
                 question,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.race_id = race_id
        self.context_sentence = context_sentence
        self.question = question
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.race_id}",
            f"article: {self.context_sentence}",
            f"question: {self.question}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
            f"option_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    """ RACE 数据集输入特征 
    Args:
        example_id: 样本id
        choice_features: 特征 ['input_ids', 'input_mask', 'segment_ids']
        label: 标签
    """
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

def save_features(dst,features):
    with open(dst,'w') as f:
        json.dump(features,f)

def load_features(src):
    with open(src,'r') as f:
        features=json.load(f)
    return features

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s.
    Args:
        examples: RACEExamples 类型
        tokenizer: 分词器
        max_seq_length: 句子最大长度
    Returns:
        features:{
            example_id: 试题id
            choices_features :[
                {article + question + option1}, ... , {article + question + option2}
            ]
            label: 标签
        }
    """
    # if(os.path.exists('.bertcache/features.json')):
    #     print('load data from cached files')
    #     return load_features('.bertcache/features.json')

    features = []
    cnt=0
    for example in tqdm.tqdm(examples,leave=False):
        question_tokens = tokenizer.tokenize(example.question) # question
        context_tokens = tokenizer.tokenize(example.context_sentence) # article

        choices_features = []
        for ending_index, ending in enumerate(example.endings):

            context_tokens_choice = context_tokens[:]
            ending_tokens = question_tokens + tokenizer.tokenize(ending) # question + option

            # "- 3" 指的是输入中包含 [CLS], [SEP], [SEP]: [CLS] Article [SEP] Question + Option [SEP]
            _truncate_seq_pair(context_tokens_choice,
                               ending_tokens, max_seq_length - 3)
            tokens = ["[CLS]"] + context_tokens_choice + \
                ["[SEP]"] + ending_tokens + ["[SEP]"]

            segment_ids = [0] * (len(context_tokens_choice) + 2) + \
                [1] * (len(ending_tokens) + 1) # article 为 0 ， question + option 为1

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append(
                (tokens, input_ids, input_mask, segment_ids))

        label = example.label

        features.append(
            InputFeatures(
                example_id=example.race_id,
                choices_features=choices_features,
                label=label
            )
        )
    # save_features('.bertcache/features.json',features)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def convert_features_to_tensors(features, batch_size):
    all_input_ids = torch.tensor(
        select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(
        select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(
        select_field(features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask,
                         all_segment_ids, all_label_ids)

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def read_race_examples(filename):

    examples = []

    with open(filename, 'r') as f:
        dataes = json.load(f)
        for data in dataes:
            examples.append(
                RaceExample(
                    race_id = data['race_id'],
                    context_sentence = data['article'],
                    question = data['question'],
                    ending_0 = data['option_0'],
                    ending_1 = data['option_1'],
                    ending_2 = data['option_2'],
                    ending_3 = data['option_3'],
                    label = data['label']
                 )
            )
    
    return examples


def load_data(filename, tokenizer, max_seq_length, batch_size):
    #读取json中的数据，examples类型是：list[RaceExample]
    examples = read_race_examples(filename)
    #转变成features
    features = convert_examples_to_features(examples, tokenizer, max_seq_length)
    dataloader = convert_features_to_tensors(features, batch_size)
    return dataloader, len(examples)

def get_device(gpu_id):
    device = torch.device("cuda:" + str(gpu_id)
                          if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu

def classifiction_metric(preds, labels, label_list):
    """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """

    acc = metrics.accuracy_score(labels, preds)

    labels_list = [i for i in range(len(label_list))]

    report = metrics.classification_report(
        labels, preds, labels=labels_list, target_names=label_list, digits=5, output_dict=True)

    return acc, report

if __name__ == "__main__":
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        "/home/songyingxin/datasets/pytorch-bert/vocabs/bert-base-uncased-vocab.txt", do_lower_case=True)

    filename = "/home/songyingxin/datasets/RACE/all/test.json"

    load_data(filename, tokenizer, 120, 32)

    print(1)
