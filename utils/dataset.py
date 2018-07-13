from torch.utils import data
import jieba
import pandas as pd
from utils import WordEmbedding
import torch
import numpy as np


class TextDataSet(data.Dataset):
    def __init__(self, path, embedding, max_seq_len=80, train=True, test=False):
        '''
        读取文件，并将句子转化成词索引
        :param train:是否训练集
        :param test: 是否测试集
        '''

        self.w2i = embedding.word2id
        self.i2w = embedding.id2word
        self.max_seq_len = max_seq_len
        data, entities, labels = self.tokenize(path, self.w2i)
        data_len = len(data)
        if test:
            self.data = data[int(0.7 * data_len):]
            self.entities = entities[int(0.7 * data_len):]
            self.labels = labels[int(0.7 * data_len):]
        elif train:
            self.data = data[:int(0.7 * data_len)]
            self.entities = entities[:int(0.7 * data_len)]
            self.labels = labels[:int(0.7 * data_len)]
        else:
            self.data = data
            self.entities = entities
            self.labels = labels

    def pad_sequence(self, sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def tokenize(self, path, word_to_id):
        f = pd.read_csv(path, encoding='utf8', index_col=0)
        data = []
        labels = []
        entities = []
        pad_and_trunc = 'post'
        cell_to_list = lambda s: s.replace('[', '').replace(']', '').split(',')
        for row in f.iterrows():
            sentence = row[1]['sentence'].replace(' ', '')
            entity_list = cell_to_list(row[1]['entity'])
            score_list = [int(float(sc)) + 1 for sc in cell_to_list(row[1]['score'])]
            words = jieba.lcut(sentence)
            sequence = [word_to_id[w] if w in word_to_id else len(word_to_id) + 1 for w in words]
            sequence = self.pad_sequence(sequence, self.max_seq_len, dtype='int64', padding=pad_and_trunc,
                                         truncating=pad_and_trunc)
            for index, entity in enumerate(entity_list):
                entities.append(word_to_id[entity] if entity in word_to_id else len(word_to_id) + 1)
                data.append(sequence)
                labels.append(score)
        return data, entities, labels

    def __getitem__(self, index):
        item = {
            'text_raw_indices': torch.tensor(self.data[index], dtype=torch.long),
            'entity_indices': torch.tensor(self.entities[index], dtype=torch.long),
            'label': torch.tensor(self.labels[index], dtype=torch.long)
        }
        return item

    def __len__(self):
        return len(self.data)
