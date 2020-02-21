import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time

from torch import optim
from tqdm import tqdm
from Seq2Eye.model import Seq2Seq
from expression_dataset import EyeExpressionDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from functools import partial
from collate import collate_fn


torch.multiprocessing.set_sharing_strategy('file_system') # to prevent error
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def custom_loss(output, trg, alpha=0.001, beta=0.1):
    '''
    Youngwoo's loss function
    loss = mse_loss + alpha * countinuity_loss + beta * variance_loss
    predict: B x S x dim
    trg: 
    '''
    n_element = output.numel()
    # mse; output will be between 0 to 1
    mse_loss = F.mse_loss(output, trg)
    # continuity
    diff = [abs(output[:, n, :] - output[:, n-1, :]) for n in range(1, output.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    # variance
    var_loss = -torch.sum(torch.norm(output, 2, 1)) / n_element
    # custom loss
    loss = mse_loss + alpha * cont_loss + beta * var_loss

    return loss


def prepare_dataloaders(data, opt):
    validation_split = 0.2 # which is 20% of whole dataset
    random_seed = 42
    # get dataset class
    eye_expression_dataset = EyeExpressionDataset(
                                word2idx=data['lang'].word2index,
                                idx2word=data['lang'].index2word,
                                src_insts=data['src_insts'],
                                trg_ints=data['trg_insts'],)
    dataset_indicies = list(range(eye_expression_dataset.__len__()))
    split_index = int(np.floor(validation_split * eye_expression_dataset.__len__()))
    
    if opt.is_shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(dataset_indicies)
    # get train and validation dataset indicies
    train_indicies, valid_indicies = dataset_indicies[split_index:], dataset_indicies[:split_index]
    train_sampler = SubsetRandomSampler(train_indicies)
    valid_sampler = SubsetRandomSampler(valid_indicies)
    # get train and valid loader
    train_loader = DataLoader(
                        dataset=eye_expression_dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        sampler=train_sampler,
                        shuffle=opt.is_shuffle,
                        collate_fn=partial(collate_fn, opt=opt))
    valid_loader = DataLoader(
                        dataset=eye_expression_dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        sampler=valid_sampler,
                        shuffle=opt.is_shuffle,
                        collate_fn=partial(collate_fn, opt=opt))

    return train_loader, valid_loader


def train(model, train_data, valid_data, optim, device, opt, start_i=0):
    for epoch_i in range(start_i, opt.epoch):
        print('[INFO] Epoch: {}'. format(epoch_i))
        # train
        start = time.time()
        train_loss = train_epoch(model, train_data, optim, device)
        print('\t- (Training)    loss: {:8.5f}, elapse: {:3.3f}'.format(
                                        train_loss, (time.time() - start)/60))
        # valid
        start = time.time()
        valid_loss = valid_epoch(model, valid_data, device)
        print('\t- (Validation)    loss: {:8.5f}, elapse: {:3.3f}'.format(
                                        valid_loss, (time.time() - start)/60))
        
        
def train_epoch(model, train_data, optim, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_data, mininterval=2, desc=' - (Training)', leave=False):
        batch_loss = 0
        n_motions = 0
        for src_seq, src_len, trg_seq in batch:
            optim.zero_grad()
            src_seq = src_seq.to(device)
            trg_seq = trg_seq.to(device)
            # model forward 
            output = model(src_seq, src_len, trg_seq)
            loss = custom_loss(output, trg_seq)
            loss.backward() # backward pass
        # optimize step
        optim.step()
        batch_loss += loss.item()
        n_motions += 1
    # calculate total loss
    total_loss += batch_loss/n_motions

    return total_loss


def valid_epoch(model, valid_data, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_data, mininterval=2, desc='  - (Validation)', leave=False):
            batch_loss = 0
            n_motions = 0
            for src_seq, src_len, trg_seq in batch:
                src_seq = src_seq.to(device)
                trg_seq = trg_seq.to(device)
                # model forward
                output = model(src_seq, src_len, trg_seq)
                loss = custom_loss(output, trg_seq)
                n_motions += 1
            total_loss += batch_loss/n_motions

    return total_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='./processed/processed_final.pickle')
    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-is_shuffle', type=bool, default=False)

    parser.add_argument('-hidden', type=int, default=200)
    parser.add_argument('-bidirectional', type=bool, default=True)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-lr', type=int, default=0.00001)
    parser.add_argument('-epoch', type=int, default=100)
    
    # for loss calculation
    # parser.add_argument('-alpha', type=int, default=0.001)
    # parser.add_argument('-beta', type=int, default=0.1)

    opt = parser.parse_args()

    # device, here we use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load dataset
    data = torch.load(opt.data)
    train_data, valid_data = prepare_dataloaders(data, opt)
    # prepare model
    print('[INFO] Preparing seq2seq model.')
    model = Seq2Seq(
                hidden=opt.hidden,
                bidirectional=opt.bidirectional,
                n_layers=opt.n_layers,
                dropout=opt.dropout,
                pre_trained_embedding=data['emb_table'],
                trg_dim=data['estimator'].n_components)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # train process
    train(model, train_data, valid_data, optimizer, device, opt)

    




    ########################## test purpose ##########################
    # dummpy src and trg data
    # src = torch.ones(2,8).long() # B x S
    # trg = torch.randn(2,30,10).float() # B x S x dim
    # o = model(src, torch.tensor([8, 8]), trg)
    # print('TEST')

if __name__ == '__main__':
    main()