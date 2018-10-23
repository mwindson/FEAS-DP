from torch.utils import data
import jieba
import pandas as pd
from utils import WordEmbedding
import torch
import numpy as np
import ast


class TextDataSet(data.Dataset):
    def __init__(self, path, word_embedding, char_embedding, max_seq_len=80, vector_level='word', train=False,
                 test=False):
        '''
        读取文件，并将句子转化成词索引
        :param train:是否训练集
        :param test: 是否测试集
        '''

        self.w2i = word_embedding.word2id
        self.i2w = word_embedding.id2word
        self.c2i = char_embedding.word2id
        self.i2c = char_embedding.id2word
        self.max_seq_len = max_seq_len
        data = self.tokenize(path, vector_level)
        data_len = len(data)
        if test:
            self.data = data[int(0.7 * data_len):]
        elif train:
            self.data = data[:int(0.7 * data_len)]
        else:
            self.data = data

    def pad_sequence(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0.):
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

    def word_char_sequence(self, text, maxlen, max_char_len=4):
        x = [np.ones(max_char_len) * 0. for i in range(maxlen)]
        unknownidx = len(self.c2i) + 1
        words = jieba.lcut(text)
        sequence = [self.pad_sequence([self.c2i[c] if c in self.c2i else unknownidx for c in w], max_char_len) for w in
                    words]
        trunc = sequence[:maxlen]
        x[:len(trunc)] = trunc
        return np.asarray(x)

    def text_to_sequence(self, text, vector_level='word', reverse=False):
        if len(text) == 0:
            return [0]
        if vector_level == 'word':
            unknownidx = len(self.w2i) + 1
            words = jieba.lcut(text)
            sequence = [self.w2i[w] if w in self.w2i else unknownidx for w in words]
        else:
            unknownidx = len(self.c2i) + 1
            words = list(text)
            sequence = [self.c2i[w] if w in self.c2i else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return sequence

    def tokenize(self, path, vector_level='word'):
        f = pd.read_csv(path, encoding='utf8', index_col=0)
        all_data = []
        pad_and_trunc = 'post'
        f.entity = f.entity.apply(lambda s: list(ast.literal_eval(s)))
        try:
            f.score = f.score.apply(lambda s: list(ast.literal_eval(s)))
        except:
            f.score = 0
        for row in f.iterrows():
            text_raw = row[1]['sentence'].lstrip().rstrip()
            entity_list = row[1]['entity']
            # score_list = [int(float(sc)) + 1 for sc in row[1]['score']]
            for index, entity in enumerate(entity_list):
                text_raw_indices = self.text_to_sequence(text_raw, vector_level)
                text_char_indices = self.text_to_sequence(text_raw, 'char')
                text_word_char_indices = self.word_char_sequence(text_raw, self.max_seq_len)
                entity = entity_list[index]
                text_left, _, text_right = [s for s in text_raw.partition(str(entity))]
                text_left_indices = self.text_to_sequence(text_left, vector_level)
                text_right_indices = self.text_to_sequence(text_right, vector_level)
                text_raw_without_entity_indices = text_left_indices + text_right_indices
                entity_indices = self.w2i[entity] if entity in self.w2i else len(self.w2i) + 1
                entity_char_indices = self.text_to_sequence(entity, 'char')
                text_left_with_entity_indices = text_left_indices + [entity_indices]
                text_right_with_entity_indices = text_right_indices + [entity_indices]
                try:
                    label = int(row[1]['score'][index]) + 1
                except:
                    label = 0
                pad = lambda seq: self.pad_sequence(seq, self.max_seq_len, dtype='int64', padding=pad_and_trunc,
                                                    truncating=pad_and_trunc)
                data = {
                    'text_raw': text_raw,
                    'entity': entity,
                    'text_raw_indices': torch.tensor(pad(text_raw_indices), dtype=torch.long),
                    'text_char_indices': torch.tensor(pad(text_char_indices), dtype=torch.long),
                    'text_word_char_indices': torch.tensor(text_word_char_indices, dtype=torch.long),
                    'text_left_indices': torch.tensor(pad(text_left_indices), dtype=torch.long),
                    'text_left_with_entity_indices': torch.tensor(pad(text_left_with_entity_indices), dtype=torch.long),
                    'text_right_indices': torch.tensor(pad(text_right_indices), dtype=torch.long),
                    'text_raw_without_entity_indices': torch.tensor(pad(text_raw_without_entity_indices),
                                                                    dtype=torch.long),
                    'text_right_with_entity_indices':
                        torch.tensor(pad(text_right_with_entity_indices), dtype=torch.long),
                    'entity_indices': torch.tensor(entity_indices, dtype=torch.long),
                    'entity_char_indices': torch.tensor(pad(entity_char_indices), dtype=torch.long),
                    'label': torch.tensor(label, dtype=torch.long),
                }
                # entities.append(word_to_id[entity] if entity in word_to_id else len(word_to_id) + 1)
                # text_raw.append(sequence)
                # labels.append(score_list[index])
                all_data.append(data)
        return all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
