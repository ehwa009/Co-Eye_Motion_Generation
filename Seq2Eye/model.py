import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from Seq2Eye.encoder import EncoderRNN
from Seq2Eye.decoder import Generator


class Seq2Seq(nn.Module):
    
    def __init__(self, src_size=8, pre_trained_embedding=None, embbedding_size=300, 
                    n_pre_motions=10, rnn_type='GRU', hidden=200, bidirectional=True, 
                    n_layers=2, trg_dim=10, use_residual=True, dropout=0.1):
        super().__init__()
        
        self.encoder = EncoderRNN(
                            src_size=src_size,
                            embbedding_size=embbedding_size,
                            pre_trained_embedding=pre_trained_embedding,
                            hidden=hidden,
                            rnn_type=rnn_type,
                            bidirectional=bidirectional,
                            n_layers=n_layers,
                            dropout=dropout)
        self.decoder = Generator(
                            encoder=self.encoder,
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
        # initialize decoder's hidden state as encoder's last hidden state (2 x b x dim)
        dec_hid = enc_hid
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