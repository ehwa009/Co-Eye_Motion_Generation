import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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