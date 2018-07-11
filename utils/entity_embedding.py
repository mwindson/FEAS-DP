import numpy as np


class EntityEmbedding():
    def __init__(self, file_path, entity_file_path, initializer='avg'):
        """
        generate entity embedding
        :param file_path:The path of word2vec file
        :param entity_file_path:The path of entity list file
        :param initializer: {'avg', 'random'}
        """
        mat, e2i, i2e = self.get_entity2vec(file_path, entity_file_path, initializer)
        self.m = mat
        self.entity2id = e2i
        self.id2entity = i2e

    def get_entity2vec(self, file_path, entity_file_path, initializer, dim=300):
        entity2id = {}
        id2entity = {}
        id = 1
        mat = [[0 for i in range(dim)]]
        entity_list = set()
        with open(entity_file_path, encoding='utf8') as f:
            for line in f:
                line = line.replace('\n', '')
                entity_list.add(line)
        with open(file_path, encoding='utf8') as f:
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                vector = [float(v) for v in values[1:301]]
                if word not in entity2id and word in entity_list:
                    mat.append(vector)
                    entity2id[word] = id
                    id2entity[id] = word
                    id += 1
            if initializer == 'avg':
                mat.append(np.mean(mat, axis=0))
            else:
                mat.append(np.random.uniform(-0.25, 0.25, dim).tolist())
        return mat, entity2id, id2entity


if __name__ == '__main__':
    import os

    entity_embedding = EntityEmbedding(os.path.abspath('..') + '/data/word2vec/sgns.financial.word',
                                       os.path.abspath('..') + '/data/entity.txt')
    print(entity_embedding.m[-1])
