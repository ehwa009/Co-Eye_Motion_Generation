import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

    def forward(self, src, src_len=None, hidden=None):
        embedded = self.embedding(src)
        if embedded.size(1) > 1: # if batch size is bigger than 1, use pack and padded seq
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len)
            output, hidden = self.gru(packed, hidden) # output: B x S x dim, hidden: S x B x H
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output) # unpacked
        else:
            output, hidden = self.gru(embedded, hidden)
        
        # if self.batch_first:
        #     hidden = hidden.transpose(0,1) # S x B x H -> B x S x H
        if self.bidirectional:
            output = output[:, :, :self.hidden] + output[:, :, self.hidden:] # sum bidirectional outputs

        return output, hidden


class Attn(nn.Module):
    
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.attn = nn.Linear(self.hidden * 2, hidden)
        self.v = nn.Parameter(torch.rand(hidden)) # module parameter
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, enc_out):
        l = enc_out.size(0)
        b = enc_out.size(1)
        
        # reshape B x S x H
        H = hidden.repeat(l, 1, 1).transpose(0, 1)
        enc_out = enc_out.transpose(0, 1) 
        attn_score = self.score(H, enc_out)
        
        return F.softmax(attn_score, dim=1).unsqueeze(1)

    def score(self, hidden, enc_out):
        '''
        concat score function
        score(s_t, h_i) = vT_a tanh(Wa[s_t; h_i])
        '''
        # normalize energy by tanh activation function (0 ~ 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, enc_out], 2))) # B x S x 2H -> B x S x H
        energy = energy.transpose(2, 1) # B x H x S
        v = self.v.repeat(enc_out.data.shape[0], 1).unsqueeze(1) # B x 1 x H
        energy = torch.bmm(v, energy) # B x 1 x S
        return energy.squeeze(1) # B x S


class BahdanauAttnDecoderRNN(nn.Module):
    
    def __init__(self, hidden=200, trg_dim=15, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.trg_dim = trg_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.attn = Attn(hidden)
        self.pre = nn.Sequential(
            nn.Linear(trg_dim + hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(hidden, hidden, n_layers, dropout=dropout)
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

        # post-linear layer
        post_out = self.post(rnn_out.squeeze(0)) # 1 x B x dim -> B x dim

        return post_out, hidden, attn_weights


class Generator(nn.Module):
    
    def __init__(self, hidden=200, trg_dim=10, n_layers=2, dropout=0.1, use_residual=True):
        super().__init__()
        self.n_layers = n_layers
        self.use_residual = use_residual
        self.decoder = BahdanauAttnDecoderRNN(hidden, trg_dim, n_layers, dropout)

    def forward(self, trg, last_hidden, enc_out):
        if self.use_residual:
            output, hid, attn = self.decoder(trg, last_hidden, enc_out)
            output = trg + output # residual connection
        else:
            output, hid, attn = self.decoder(trg, last_hidden, enc_out)
        
        return output, hid, attn


class Seq2Seq(nn.Module):
    
    def __init__(self, src_size=8, pre_trained_embedding=None, embbedding_size=300, 
                    n_pre_motions=10, hidden=200, bidirectional=True, 
                    n_layers=2, trg_dim=10, use_residual=True, dropout=0.1):
        super().__init__()
        
        self.encoder = EncoderRNN(
                            src_size=src_size,
                            embbedding_size=embbedding_size,
                            pre_trained_embedding=pre_trained_embedding,
                            hidden=hidden,
                            bidirectional=bidirectional,
                            n_layers=n_layers,
                            dropout=dropout)
        self.decoder = Generator(
                            hidden=hidden,
                            trg_dim=trg_dim,
                            n_layers=n_layers,
                            dropout=dropout,
                            use_residual=use_residual)
        self.n_pre_motions = n_pre_motions

    def forward(self, src, src_len, trg):
        # reshape to S x B x dim
        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)
        # run words through the encoder
        enc_out, enc_hid = self.encoder(src, src_len)
        dec_hid = enc_hid[:self.decoder.n_layers]
        # set output to be stored
        all_dec_out = torch.zeros(trg.size(0), trg.size(1), trg.size(2)).to(trg.device) # B x S x dim
        
        # run through decoder one time step at a time
        dec_in = trg[0] # set inital motion
        all_dec_out[0] = dec_in
        for step in range(1, trg.size(0)):
            dec_out, dec_hid, _ = self.decoder(dec_in, dec_hid, enc_out)
            all_dec_out[step] = dec_out
            if step < self.n_pre_motions: # use teacher forcing until n-previous motions
                dec_in = trg[step]
            else:
                dec_in = dec_out

        return all_dec_out.transpose(0, 1)


# if __name__ == '__main__':
    
#     # dummpy src and trg data
#     src = torch.ones(2,8).long()
#     trg = torch.randn(2,30,10).float()
    
#     model = Seq2Seq()
    
#     o = model(src, torch.tensor([8, 8]), trg)