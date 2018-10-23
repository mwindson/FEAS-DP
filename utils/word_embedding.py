import numpy as np
import os


class WordEmbedding():
    def __init__(self, file_path, initializer="random"):
        mat, w2i, i2w = self.load_word2vec(file_path, initializer=initializer)
        self.m = mat
        self.word2id = w2i
        self.id2word = i2w

    def load_word2vec(self, file_path, dim=300, initializer="random"):
        word2id = {}
        id2word = {}
        id = 1
        mat = [[0 for i in range(dim)]]
        entity_list = set()
        entity_mat = []
        with open(os.path.abspath('.') + '/data/entity.txt', encoding='utf8') as f:
            for line in f:
                line = line.replace('\n', '')
                entity_list.add(line)
        with open(file_path, encoding='utf8') as f:
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                vector = [float(v) for v in values[1:301]]
                if word not in word2id:
                    mat.append(vector)
                    word2id[word] = id
                    id2word[id] = word
                    id += 1
                if word in entity_list:
                    entity_mat.append(vector)
            if initializer == 'avg':
                mat.append(np.mean(entity_mat, axis=0))
            else:
                mat.append(np.random.uniform(-0.25, 0.25, dim).tolist())
        return mat, word2id, id2word
