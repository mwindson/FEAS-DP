import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
from layers.word_char_embedding import WordCharEmbedding


class LSTM_CNN(nn.Module):
    def __init__(self, word_embedding, char_embedding, opt):
        super(LSTM_CNN, self).__init__()
        self.opt = opt
        self.word_char_embed = WordCharEmbedding(word_embedding, char_embedding, opt)
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.bi_lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                          bidirectional=True)
        D = opt.embed_dim
        C = opt.polarities_dim
        # A = opt.aspect_num
        Co = 100
        # Ks = [i for i in range(3, 10)]
        Ks = [3, 4]
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K - 2) for K in [3]])
        self.fc_aspect = nn.Linear(100, Co)
        self.attention = Attention(opt.hidden_dim * 2, score_function='mlp')
        self.l1 = nn.Linear(opt.hidden_dim * 2, Co * len(Ks))
        self.dropout = nn.Dropout(opt.dropout)
        # self.dense = nn.Linear(self.kernel_num, opt.polarities_dim)
        self.gru_cell = nn.GRUCell(opt.hidden_dim * 2, opt.hidden_dim * 2)
        self.gru = nn.GRUCell(Co * len(Ks), Co * len(Ks))
        self.dense = nn.Linear(Co * len(Ks) * 2, opt.polarities_dim)
        self.w1 = nn.Linear(Co * len(Ks), Co * len(Ks))
        self.w2 = nn.Linear(Co * len(Ks), Co * len(Ks))

    def get_position_embedding(self, len):
        mat = None
        dim = 100
        for n_position in len.cpu().numpy():
            x = (np.zeros((self.opt.max_seq_len, dim)) * 0.).astype(dtype='float64')

            def cal_angle(position, hid_idx):
                return position / np.power(10000, 2 * (hid_idx // 2) / dim)

            def get_posi_angle_vec(position):
                return [cal_angle(position, hid_j) for hid_j in range(dim)]

            sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
            x[:n_position, :] = sinusoid_table
            embed = torch.FloatTensor(x).unsqueeze(0)
            if mat is None:
                mat = embed
            else:
                mat = torch.cat((mat, embed), 0)
        return mat

    def gate(self, s1, s2):
        z1 = self.w1(s1)
        z2 = self.w2(s2)

        zz = torch.cat([torch.unsqueeze(z1, 0), torch.unsqueeze(z2, 0)], 0)
        zz = F.softmax(zz, 0)
        z1 = torch.squeeze(zz[0], 0)
        z1 = torch.squeeze(zz[1], 0)

        out = torch.mul(z1, s1) + torch.mul(z2, s2)
        return out

    def forward(self, inputs):
        text_raw_indices, aspect_indices, text_word_char_indices, aspect_char_indices = inputs[0], inputs[1], inputs[2], \
                                                                                        inputs[3]
        # embed
        text_embed, aspect_embed = self.word_char_embed(text_raw_indices, text_word_char_indices, aspect_char_indices)
        # text
        text_len = torch.sum(text_raw_indices != 0, dim=-1)
        # aspect
        if len(aspect_indices.shape) == 1:
            aspect_indices = torch.unsqueeze(aspect_indices, dim=1)
        aspect_len = torch.sum(aspect_char_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        # lstm
        memory, (_, _) = self.text_lstm(text_embed, text_len)
        aspect, (_, _) = self.bi_lstm_aspect(aspect_embed, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        # attention
        et = aspect
        for _ in range(self.opt.hops):
            it_al = self.attention(memory, et).squeeze(dim=1)
            et = self.gru_cell(it_al, et)
        et = self.l1(et)
        # cnn
        aa = [F.relu(conv(aspect_embed.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_embed = torch.cat(aa, 1)

        x = [F.tanh(conv(text_embed.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        # y = [F.relu(conv(text_embed.transpose(1, 2)) + self.fc_aspect(aspect_embed).unsqueeze(2)) for conv in
        #      self.convs2]
        # x = [i * j for i, j in zip(x, y)]

        # pooling method
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        # output
        out = self.dense(self.dropout(torch.cat([x, et], 1)))
        # out = self.dense(memory)
        return F.softmax(out, dim=1)
