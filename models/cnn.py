# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CNN, self).__init__()
        self.num = 10
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
        self.con1v = nn.Sequential(   #(1,80,300)
            nn.Conv1d(1, 100, 3*300,300),
            nn.ReLU(),
            nn.MaxPool1d(78),
        )  #( 100,1,150)
        for i in range(3,self.num):
            conv = nn.Sequential(nn.Conv1d(1,100,i*300,300), nn.ReLU(), nn.MaxPool1d(80-i+1))
            setattr(self,f'con_{i}',conv)
        self.out = nn.Linear((self.num-3)*100,opt.polarities_dim)
        #self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        # self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    def get_conv(self, i):
        return getattr(self, f'con_{i}')

    def forward(self, inputs):

        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices).view(-1,1,80*300)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        #X = self.con1v(x)
        X = [self.get_conv(i)(x) for i in range(3,self.num)]
        X = torch.cat(X,1)
        #X = self.con2v(X)
        X = X.view(X.size(0),-1)
        output = self.out(X)
        return output