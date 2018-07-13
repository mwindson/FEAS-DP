from torch import nn
from layers.attention import Attention
from layers.dynamic_rnn import DynamicLSTM
import torch


class ATAE_LSTM(nn.Module):
    def __init__(self, embedding, opt):
        super(ATAE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))
        self.max_seq_len = opt.max_seq_len
        self.lstm = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = Attention(opt.hidden_dim, out_dim=opt.polarities_dim,
                                   score_function='mlp')

    def forward(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        x = self.embed(text_indices)
        aspect = self.embed(aspect_indices)
        x_len = torch.sum(text_indices != 0, dim=-1)
        aspects = torch.unsqueeze(aspect, 1).repeat(1, self.max_seq_len, 1)
        x = torch.cat((x, aspects), 2)  # aspect embedding 拼接到 word embedding后
        lstm_out, (h_n, _) = self.lstm(x, x_len)
        out = self.attention(lstm_out, aspect)
        return torch.squeeze(out)
