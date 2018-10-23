import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, embedding, opt):
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.opt = opt

        # V = opt.embed_num
        D = opt.embed_dim
        C = opt.polarities_dim
        # A = opt.aspect_num

        Co = 100
        Ks = [3, 4]

        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding))

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K - 2) for K in [3]])

        # self.convs3 = nn.Conv1d(D, 300, 3, padding=1), smaller is better
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc_aspect = nn.Linear(100, Co)

    def forward(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]
        if len(aspect_indices.shape) == 1:
            aspect_indices = torch.unsqueeze(aspect_indices, dim=1)
        feature = self.embed(text_indices)  # (N, L, D)
        aspect_v = self.embed(aspect_indices)  # (N, L', D)
        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        # aa = F.tanhshrink(self.convs3(aspect_v.transpose(1, 2)))  # [(N,Co,L), ...]*len(Ks)
        # aa = F.max_pool1d(aa, aa.size(2)).squeeze(2)
        # aspect_v = aa
        # smaller is better

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]

        # pooling method
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        # x = [F.adaptive_max_pool1d(i, 2) for i in x]
        # x = [i.view(i.size(0), -1) for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc1(x)  # (N,C)
        return logit
