import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

'''
GRU
'''
class EncoderRNN(nn.Module):
    
    def __init__(self, src_size, embbedding_size, pre_trained_embedding=None, hidden=200, bidirectional=True, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout = dropout

        if pre_trained_embedding is not None:
            # get embedding layer - glove
            self.embedding = nn.Embedding.from_pretrained(
                                torch.from_numpy(pre_trained_embedding).float(),
                                freeze=True)
        else:
            self.embedding = nn.Embedding(src_size, embbedding_size)

        self.gru = nn.GRU(embbedding_size,
                        hidden,
                        bidirectional=self.bidirectional,
                        num_layers=n_layers)

        # self.lstm = nn.LSTM(embbedding_size, hidden, n_layers, 
        #                     bidirectional=bidirectional,
        #                     dropout=dropout)

    def forward(self, src, src_len=None, hidden=None):
        embedded = self.embedding(src)
        if embedded.size(1) > 1: # if batch size is bigger than 1, use pack and padded seq
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len)
            output, hidden = self.gru(packed, hidden) # output: B x S x dim, hidden: S x B x H
            # output, hidden = self.lstm(packed, hidden)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output) # unpacked
        else:
            output, hidden = self.gru(embedded, hidden)
            # _, output, hidden = self.lstm(embedded, hidden)
        
        # if self.batch_first:
        #     hidden = hidden.transpose(0,1) # S x B x H -> B x S x H
        if self.bidirectional:
            output = output[:, :, :self.hidden] + output[:, :, self.hidden:] # sum bidirectional outputs

        return output, hidden

'''
LSTM
'''
class EncoderLSTM(nn.Module):
    
    def __init__(self, src_size, embbedding_size, pre_trained_embedding=None, hidden=200, bidirectional=True, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout = dropout

        if pre_trained_embedding is not None:
            # get embedding layer - glove
            self.embedding = nn.Embedding.from_pretrained(
                                torch.from_numpy(pre_trained_embedding).float(),
                                freeze=True)
        else:
            self.embedding = nn.Embedding(src_size, embbedding_size)

        self.lstm = nn.LSTM(embbedding_size, hidden, n_layers, 
                            bidirectional=bidirectional,
                            dropout=dropout)

    def forward(self, src, src_len=None, hidden=None):
        embedded = self.embedding(src)
        if embedded.size(1) > 1: # if batch size is bigger than 1, use pack and padded seq
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len)
            packed_output, (hidden, cell) = self.lstm(packed_input, hidden)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output) # unpacked
        else:
            output, (hidden, cell) = self.lstm(embedded, hidden)
        
        # if self.batch_first:
        #     hidden = hidden.transpose(0,1) # S x B x H -> B x S x H
        if self.bidirectional:
            hidden = hidden[:, :, :self.hidden] + hidden[:, :, self.hidden:]
            cell = cell[:, :, :self.hidden] + cell[:, :, self.hidden:]

        return hidden, cell