from torch import nn
from layers.attention import Attention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn.functional as F
import torch.nn.init as init


class LRG(nn.Module):
    def __init__(self, embedding, opt):
        super(LRG, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))
        self.bi_all_ctx = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                      bidirectional=True)
        self.bi_left_ctx = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                       bidirectional=True)
        self.bi_right_ctx = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                        bidirectional=True)
        self.bi_lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                          bidirectional=True)
        self.attention = Attention(opt.hidden_dim * 2, score_function='mlp')
        self.left_attention = Attention(opt.hidden_dim * 2, score_function='mlp')
        self.right_attention = Attention(opt.hidden_dim * 2, score_function='mlp')
        self.left_gru_cell = nn.GRUCell(opt.hidden_dim * 2, opt.hidden_dim * 2)
        self.right_gru_cell = nn.GRUCell(opt.hidden_dim * 2, opt.hidden_dim * 2)
        self.w1 = nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2)
        self.w2 = nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2)
        self.w3 = nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def gate(self, s, sl, sr, ht):
        # input = torch.cat((sl, sr), 1)  # (batch_size,embed_dim * 2)
        # g = F.softmax(self.w_g(input))  # (batch_size,1)
        #
        z = self.w3(s)
        zl = self.w1(sl)
        zr = self.w2(sr)

        zz = torch.cat([torch.unsqueeze(zl, 0), torch.unsqueeze(zr, 0), torch.unsqueeze(z, 0)], 0)
        zz = F.softmax(zz, 0)
        z = zz[2].squeeze(0)
        zl = torch.squeeze(zz[0], 0)
        zr = torch.squeeze(zz[1], 0)

        out = torch.mul(zl, sl) + torch.mul(zr, sr) + torch.mul(z, s)
        return out

    def forward(self, inputs):
        # inputs
        text_raw_indices, aspect_indices, text_left, text_right = inputs[0], inputs[1], inputs[2], inputs[3]
        # aspect
        if len(aspect_indices.shape) == 1:
            aspect_indices = torch.unsqueeze(aspect_indices, dim=1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect = self.embed(aspect_indices)
        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        # all
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        x_mem, (_, _) = self.bi_all_ctx(x, x_len)
        all = self.attention(x_mem, aspect).squeeze(dim=1)
        # left memory
        x_left = self.embed(text_left)
        left_len = torch.sum(text_left != 0, dim=-1)
        left_mem, (_, _) = self.bi_left_ctx(x_left, left_len)
        left = aspect
        # for _ in range(self.opt.hops):
        left = self.left_attention(left_mem, left).squeeze(dim=1)
        # left = self.left_gru_cell(it_al, left)
        # right memory
        x_right = self.embed(text_right)
        right_len = torch.sum(text_right != 0, dim=-1)
        right_mem, (_, _) = self.bi_right_ctx(x_right, right_len)
        right = aspect
        # for _ in range(self.opt.hops):
        right = self.right_attention(right_mem, right).squeeze(dim=1)
        # right = self.right_gru_cell(it_al, right)
        out = self.gate(all, left, right, aspect)
        out = self.dense(out)
        return out
