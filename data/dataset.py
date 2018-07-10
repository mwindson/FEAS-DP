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
        data, labels = self.tokenize(path, self.w2i)
        data_len = len(data)
        if test:
            self.data = data[int(0.7 * data_len):]
            self.labels = labels[int(0.7 * data_len):]
        elif train:
            self.data = data[:int(0.7 * data_len)]
            self.labels = labels[:int(0.7 * data_len)]
        else:
            self.data = data
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
        pad_and_trunc = 'post'
        for row in f.iterrows():
            sentence = row[1]['sentence'].replace(' ', '')
            words = jieba.lcut(sentence)
            sequence = [word_to_id[w] if w in word_to_id else len(word_to_id) + 1 for w in words]
            sequence = self.pad_sequence(sequence, self.max_seq_len, dtype='int64', padding=pad_and_trunc,
                                         truncating=pad_and_trunc)
            data.append(sequence)
            labels.append(row[1]['score'] + 1)
        return data, labels

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.long)
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        return data, labels

    def __len__(self):
        return len(self.data)
