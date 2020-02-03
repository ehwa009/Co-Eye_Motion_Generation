import argparse
import torch
import torch.nn.functional as F

from torch import optim
from tqdm import tqdm
from Seq2Eye.model import Seq2Seq

torch.multiprocessing.set_sharing_strategy('file_system') # to prevent error
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def custom_loss(predict, trg, alpha, beta):
    '''
    Youngwoo's loss function
    loss = mse_loss + alpha * countinuity_loss + beta * variance_loss
    predict: B x S x dim
    trg: 
    '''
    n_element = predict.numel()
    
    # mse; output will be between 0 to 1
    mse_loss = F.mse_loss(predict, trg)

    # continuity
    diff = [abs(predict[:, n, :] - predict[:, n-1, :]) for n in range(1, predict.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element

    # variance
    var_loss = -torch.sum(torch.norm(predict, 2, 1)) / n_element

    # custom loss
    loss = mse_loss + alpha * cont_loss + beta * var_loss

    return loss





def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('-data', default='./processed/preprocessing.pickle')

    parser.add_argument('-hidden', type=int, default=200)
    parser.add_argument('-bidirectional', type=bool, default=True)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-dropout', type=int, default=0.1)
    opt = parser.parse_args()

    # load data
    # opt.src_size = 


    # prepare model
    print('[INFO] preparing seq2seq model.')
    model = Seq2Seq(
                hidden=opt.hidden,
                bidirectional=opt.bidirectional,
                n_layers=opt.n_layers,
                dropout=opt.dropout
    )

    # test purpose
    # dummpy src and trg data
    src = torch.ones(2,8).long() # B x S
    trg = torch.randn(2,30,10).float() # B x S x dim
    o = model(src, torch.tensor([8, 8]), trg)
    print('TEST')

if __name__ == '__main__':
    main()