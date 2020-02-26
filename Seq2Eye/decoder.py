import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from Seq2Eye.attn import Attn

'''
GRU
'''
class BahdanauAttnDecoderRNN(nn.Module):
    
    def __init__(self, hidden=200, trg_dim=15, n_layers=2, bidirectional=True, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.trg_dim = trg_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.attn = Attn(hidden)
        self.pre = nn.Sequential(
            nn.Linear(trg_dim + hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(hidden, hidden, n_layers, 
                        bidirectional=bidirectional, dropout=dropout)
        # self.lstm = nn.LSTM(hidden, hidden, n_layers, 
        #                     dropout=dropout, bidirectional=bidirectional)
        self.post = nn.Linear(hidden, trg_dim)

    def forward(self, trg, last_hidden, enc_out):
        trg = trg.view(1, trg.size(0), -1) # 1 x B x dim
        
        # attention
        attn_weights = self.attn(last_hidden[-1], enc_out) # B x 1 x S
        context = attn_weights.bmm(enc_out.transpose(0, 1)) # B x 1 x H(attn_size)
        context = context.transpose(0, 1) # 1 x B x H(attn_size)

        # pre-linear layer
        pre_input = torch.cat((trg, context), 2) # 1 x B x (dim + attn_size)
        pre_out = self.pre(pre_input.squeeze(0))
        pre_out = pre_out.unsqueeze(0)

        # rnn layer
        rnn_out, hidden = self.gru(pre_out, last_hidden) # out: 1 x B x dim, hid: n_layer x B x H
        # rnn_out, hidden = self.lstm(pre_out, last_hidden)

        if self.bidirectional:
            rnn_out = rnn_out[:, :, :self.hidden] + rnn_out[:, :, self.hidden:]

        # post-linear layer
        post_out = self.post(rnn_out.squeeze(0)) # 1 x B x dim -> B x dim

        return post_out, hidden, attn_weights


class Generator(nn.Module):
    
    def __init__(self, hidden=200, trg_dim=10, n_layers=2, bidirectional=True, dropout=0.1, use_residual=True):
        super().__init__()
        self.n_layers = n_layers
        self.use_residual = use_residual
        self.decoder = BahdanauAttnDecoderRNN(hidden, trg_dim, n_layers, bidirectional, dropout)

    def forward(self, trg, last_hidden, enc_out):
        if self.use_residual:
            output, hid, attn = self.decoder(trg, last_hidden, enc_out)
            output = trg + output # residual connection
        else:
            output, hid, attn = self.decoder(trg, last_hidden, enc_out)
        
        return output, hid, attn

'''
LSTM
'''
class BahdanauAttnDecoderLSTM(nn.Module):
    
    def __init__(self, hidden=200, trg_dim=15, n_layers=2, bidirectional=True, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.trg_dim = trg_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.attn = Attn(hidden)
        self.pre = nn.Sequential(
            nn.Linear(trg_dim + hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True)
        )
        self.lstm = nn.LSTM(hidden, hidden, n_layers, 
                            dropout=dropout, bidirectional=bidirectional)
        self.post = nn.Linear(hidden, trg_dim)

    def forward(self, trg, last_hidden, enc_out):
        trg = trg.view(1, trg.size(0), -1) # 1 x B x dim
        
        # attention
        attn_weights = self.attn(last_hidden[-1], enc_out) # B x 1 x S
        context = attn_weights.bmm(enc_out.transpose(0, 1)) # B x 1 x H(attn_size)
        context = context.transpose(0, 1) # 1 x B x H(attn_size)

        # pre-linear layer
        pre_input = torch.cat((trg, context), 2) # 1 x B x (dim + attn_size)
        pre_out = self.pre(pre_input.squeeze(0))
        pre_out = pre_out.unsqueeze(0)

        # rnn layer
        output, (hidden, cell) = self.lstm(pre_out, last_hidden)

        if self.bidirectional:
            rnn_out = output[:, :, :self.hidden] + output[:, :, self.hidden:]

        # post-linear layer
        post_out = self.post(rnn_out.squeeze(0)) # 1 x B x dim -> B x dim

        return post_out, hidden, attn_weights


class GeneratorLSTM(nn.Module):
    
    def __init__(self, hidden=200, trg_dim=10, n_layers=2, bidirectional=True, dropout=0.1, use_residual=True):
        super().__init__()
        self.n_layers = n_layers
        self.use_residual = use_residual
        self.decoder = BahdanauAttnDecoderLSTM(hidden, trg_dim, n_layers, bidirectional, dropout)

    def forward(self, trg, last_hidden, enc_out):
        if self.use_residual:
            output, hid, attn = self.decoder(trg, last_hidden, enc_out)
            output = trg + output # residual connection
        else:
            output, hid, attn = self.decoder(trg, last_hidden, enc_out)
        
        return output, hid, attn