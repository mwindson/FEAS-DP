import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from utils import TextDataSet, WordEmbedding
import os
from models import LSTM, AT_LSTM, ATAE_LSTM, RAM, IAN, CNN, Cabasc, MemNet, LRG


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        w2v_path = {
            'word': '/data/word2vec/sgns.financial.word',
            'word_bigram': '/data/word2vec/sgns.financial.bigram',
            'char': '/data/word2vec/sgns.financial.char',
            'char_bigram': '/data/word2vec/sgns.financial.bigram-char'
        }
        embed = WordEmbedding(os.path.dirname(__file__) + w2v_path[opt.vector_level], initializer='avg')
        data_set = TextDataSet(os.path.dirname(__file__) + '/data/single_2500_temp.csv', embed,
                               max_seq_len=opt.max_seq_len, vector_level=opt.vector_level)
        self.data_loader = DataLoader(dataset=data_set, batch_size=512)


        self.model = opt.model_class(embed.m, opt).to(opt.device)
        self.model.load_state_dict(torch.load(os.path.dirname(__file__) + '/best/'+ opt.model_name + '_best.pkl')['state_dict'])


    def RAM_DP(self):

        pres = pd.DataFrame([], columns=['sentence', 'entity', 'predict'])
        for t_batch, t_sample_batched in enumerate(self.data_loader):
            t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
            t_targets = t_sample_batched['label'].to(opt.device)
            t_outputs = self.model(t_inputs)
            outputs = torch.argmax(t_outputs, -1)
            for i, d in enumerate(outputs.cpu().numpy()):
                    pre = {'sentence': t_sample_batched['text_raw'][i],
                           'entity': t_sample_batched['entity'][i],
                           'predict': outputs.cpu().numpy()[i]}
                    pres = pres.append(pre, ignore_index=True)
        pres.to_csv('./output/result.csv')












if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ram', type=str)
    parser.add_argument('--vector_level', default='word', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--input_path', default='../input/temp.csv', type=str)
    parser.add_argument('--out_path', default='../output/', type=str)
    parser.add_argument('--sen_dic_path', default='../FESA/dictionary', type=str)

    opt = parser.parse_args()


    model_classes = {
        'lstm': LSTM,
        'at_lstm': AT_LSTM,
        'atae_lstm': ATAE_LSTM,
        'cnn': CNN,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'lrg': LRG
    }

    input_colses = {
        'lstm': ['text_raw_indices'],
        'at_lstm': ['text_raw_indices', 'entity_indices'],
        'atae_lstm': ['text_raw_indices', 'entity_indices'],
        'cnn': ['text_raw_indices'],
        'ian': ['text_raw_indices', 'entity_indices'],
        'memnet': ['text_raw_without_entity_indices', 'entity_indices', 'text_left_with_entity_indices'],
        'ram': ['text_raw_indices', 'entity_indices'],
        'cabasc': ['text_raw_indices', 'entity_indices', 'text_left_with_entity_indices',
                   'text_right_with_entity_indices'],
        'lrg': ['text_raw_indices', 'entity_indices', 'text_left_with_entity_indices',
                'text_right_with_entity_indices'],
    }

    # f = ins.FESA()
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.RAM_DP()