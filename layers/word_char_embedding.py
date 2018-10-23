import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM


class WordCharEmbedding(nn.Module):
    def __init__(self, word_embedding, char_embedding, opt):
        super(WordCharEmbedding, self).__init__()
        self.opt = opt
        self.w_embed = nn.Embedding.from_pretrained(torch.tensor(word_embedding, dtype=torch.float))
        self.c_embed = nn.Embedding.from_pretrained(torch.tensor(char_embedding, dtype=torch.float))
        D = opt.embed_dim
        Co = 300
        Ks = [2, 3]
        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.char_lstm = DynamicLSTM(opt.embed_dim, opt.embed_dim, num_layers=1, batch_first=True,
                                     only_use_last_hidden_state=True)
        self.w1 = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.w2 = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim * 2, opt.embed_dim)

    def gate(self, s1, s2):
        z1 = self.w1(s1)
        z2 = self.w2(s2)

        zz = torch.cat([torch.unsqueeze(z1, 0), torch.unsqueeze(z2, 0)], 0)
        zz = F.softmax(zz, 0)
        z1 = torch.squeeze(zz[0], 0)
        z1 = torch.squeeze(zz[1], 0)

        out = torch.mul(z1, s1) + torch.mul(z2, s2)
        return out

    def forward(self, word_indices, word_char_indices, aspect_char_indices):
        # word_char_indices (batch,len,word_len)
        indices_len = torch.sum(word_char_indices != 0, dim=-1)
        char_embeds = self.c_embed(word_char_indices)
        word_embeds = self.w_embed(word_indices)
        mixed_embed = []
        for i in range(char_embeds.shape[0]):
            char_embed = char_embeds[i]
            len = torch.sum(word_char_indices[i] != 0, dim=-1)
            len = torch.add(len, 1, (len == 0).long())
            # lstm
            x = self.char_lstm(char_embed, len)
            # cnn
            # conv = [F.relu(conv(char_embed.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
            # conv = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in conv]
            # x = torch.cat(conv, 1)
            word_embed = word_embeds[i]
            mixed = self.gate(x.squeeze(0), word_embed)
            mixed_embed.append(mixed.unsqueeze(0))
        gated_word_embed = torch.cat(mixed_embed, 0)
        aspect_embed = self.c_embed(aspect_char_indices)
        return gated_word_embed, aspect_embed
