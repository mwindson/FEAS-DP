from torch import nn
from layers.attention import Attention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn.functional as F


class LRG(nn.Module):
    def __init__(self, embedding, opt):
        super(LRG, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = Attention(opt.hidden_dim, out_dim=opt.polarities_dim,
                                   score_function='mlp')
        self.w_g = nn.Linear(opt.embed_dim * 2, 1)

    def gate(self, left_context, right_context):
        input = torch.cat((left_context, right_context), 2)  # (batch_size,len,embed_dim)
        g = F.sigmoid(self.w_g(input))  # (batch_size,1)
        out = g * left_context + (1 - g) * right_context
        return out

    def forward(self, inputs):
        # inputs
        text_raw_indices, aspect_indices, x_l, x_r = inputs[0], inputs[1], inputs[2], inputs[3]
        aspect = self.embed(aspect_indices)
        left_ctx = self.embed(x_l)
        right_ctx = self.embed(x_r)
        x = self.gate(left_ctx, right_ctx)
        x_sum = torch.sum(x, dim=2)
        x_len = torch.sum(x_sum != 0, dim=-1)
        lstm_out, (h_n, _) = self.lstm(x, x_len)
        out = self.attention(lstm_out, aspect)
        return torch.squeeze(out)
