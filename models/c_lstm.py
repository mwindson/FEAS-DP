from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F


class C_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(C_LSTM, self).__init__()
        self.opt = opt
        self.kernel_num = 100
        self.kernel_sizes = [3, 4, 5]
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
        self.context_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                        bidirectional=True)
        self.bi_lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                          bidirectional=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(opt.embed_dim, self.kernel_num, K) for K in self.kernel_sizes])
        self.attention = Attention(opt.hidden_dim * 2, score_function='mlp')
        self.dropout = nn.Dropout(opt.dropout)
        self.gru_cell = nn.GRUCell(opt.hidden_dim * 2, opt.hidden_dim * 2)
        # self.dense = nn.Linear(self.kernel_num, opt.polarities_dim)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        # text cnn
        text_embed = self.embed(text_raw_indices)
        text_len = torch.sum(text_raw_indices != 0, dim=-1)
        text_conved = [F.relu(conv(text_embed.transpose(1, 2))) for conv in self.convs1]
        memory, (_, _) = self.context_lstm(x, text_len)
        # aspect
        if len(aspect_indices.shape) == 1:
            aspect_indices = torch.unsqueeze(aspect_indices, dim=1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect = self.embed(aspect_indices)
        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))

        # attention
        et = self.attention(memory, aspect).squeeze(1)
        # output
        out = self.dense(self.dropout(et))
        # out = self.dense(memory)
        return out
