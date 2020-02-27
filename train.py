import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
import os

from torch import optim
from tqdm import tqdm, tqdm_gui
from Seq2Eye.model import Seq2Seq
from expression_dataset import EyeExpressionDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
# from functools import partial
from constant import *
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
    # diff = [abs(output[:, n, :] - output[:, n-1, :]) for n in range(1, output.shape[1])]
    # cont_loss = torch.sum(torch.stack(diff)) / n_element
    # variance
    var_loss = -torch.sum(torch.norm(output, 2, 1)) / n_element
    # custom loss
    # loss = mse_loss + alpha * cont_loss + beta * var_loss
    loss = mse_loss + beta * var_loss

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
                        collate_fn=collate_fn)
    valid_loader = DataLoader(
                        dataset=eye_expression_dataset,
                        batch_size=opt.batch_size,
                        num_workers=opt.num_workers,
                        sampler=valid_sampler,
                        collate_fn=collate_fn)

    return train_loader, valid_loader


def train(model, train_data, valid_data, optim, device, opt, start_i):
    if opt.log:
        log_train_file = opt.log + '/train.log'
        log_valid_file = opt.log + '/valid.log'
        print('[INFO] Training performance will be written to {} and {}'.format(log_train_file, log_valid_file))
        # check log file exists or not
        if not(os.path.exists(opt.log)):
            os.mkdir(opt.log)
        if not(os.path.exists(log_train_file) and os.path.exists(log_valid_file)):
            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss\n')
                log_vf.write('epoch,loss\n')
        
    train_loss_list = []
    valid_loss_list = []
    for epoch_i in tqdm_gui(range(start_i, opt.epoch)):
        print('[INFO] Epoch: {}'. format(epoch_i))
        # train process
        start = time.time()
        train_loss = train_epoch(model, train_data, optim, device, opt)
        print('\t- (Training)    loss: {:8.5f}, elapse: {:3.3f}'.format(
                                        train_loss, (time.time() - start)/60))
        train_loss_list += [train_loss] # record each train loss
        # valid process
        start = time.time()
        valid_loss = valid_epoch(model, valid_data, device, opt)
        print('\t- (Validation)    loss: {:8.5f}, elapse: {:3.3f}'.format(
                                        valid_loss, (time.time() - start)/60))
        valid_loss_list += [valid_loss] # record each valid loss

        # record train and valid log files
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{},{:8.5f}\n'.format(epoch_i, train_loss))
                log_vf.write('{},{:8.5f}\n'.format(epoch_i, valid_loss))

        # to save trained model
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'setting': opt,
            'epoch': epoch_i
        }
        if not(os.path.exists(opt.chkpt)):
            os.mkdir(opt.chkpt)
        if opt.save_mode == 'best':
            model_name = '{}/eye_model.chkpt'.format(opt.chkpt)
            if train_loss <= min(train_loss_list):
                torch.save(checkpoint, model_name)
                print('\t[INFO] The checkpoint has been updated ({}).'.format(opt.save_mode))
        elif opt.save_mode == 'interval':
            if (epoch_i % opt.save_interval) == 0 and epoch_i != 0:
                model_name = '{}/{}_{:0.3f}.chkpt'.format(opt.chkpt, epoch_i, train_loss)
                torch.save(checkpoint, model_name)
                print('\t[INFO] The checkpoint has been updated ({}).'.format(opt.save_mode))
        elif opt.save_mode == 'best_and_interval':
            model_name = '{}/eye_model.chkpt'.format(opt.chkpt)
            if train_loss <= min(train_loss_list):
                torch.save(checkpoint, model_name)
                print('\t[INFO] The best has been updated ({}).'.format(opt.save_mode))
            if (epoch_i % opt.save_interval) == 0 and epoch_i != 0:
                model_name = '{}/{}_{:0.3f}.chkpt'.format(opt.chkpt, epoch_i, train_loss)
                torch.save(checkpoint, model_name)
                print('\t[INFO] The checkpoint has been saved ({}).'.format(opt.save_mode))
        # save last trained model
        if epoch_i == (opt.epoch - 1):
            model_name = '{}/{}_{:0.3f}.chkpt'.format(opt.chkpt, epoch_i, train_loss)
            torch.save(checkpoint, model_name)
            print('\t[INFO] The last checkpoint has been saved.')
        
        
def train_epoch(model, train_data, optim, device, opt):
    model.train()
    total_loss = 0
    for batch in tqdm(train_data, mininterval=2, desc=' - (Training)', leave=False):
        mini_batch_loss = 0
        for src_seq, src_len, trg_seq in batch:
            # make zero gradient
            optim.zero_grad() 
            # model forward 
            output = model(src_seq.to(device), src_len, trg_seq.to(device))
            loss = custom_loss(output, trg_seq.to(device), opt.alpha, opt.beta)
            # backward pass
            loss.backward() 
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            # optimize step
            optim.step()
            # calculate total loss
            mini_batch_loss += loss.item()
        total_loss += mini_batch_loss
    
    return total_loss / len(train_data)


def valid_epoch(model, valid_data, device, opt):
    model.eval()
    with torch.no_grad():  
        total_loss = 0
        for batch in tqdm(valid_data, mininterval=2, desc='  - (Validation)', leave=False):
            mini_batch_loss = 0
            for src_seq, src_len, trg_seq in batch:
                # model forward
                output = model(src_seq.to(device), src_len, trg_seq.to(device))
                loss = custom_loss(output, trg_seq.to(device), opt.alpha, opt.beta)
                mini_batch_loss += loss.item()
            total_loss += mini_batch_loss

    return total_loss / len(valid_data)


def main():
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument('-data', default='./processed/processed_final.pickle')
    parser.add_argument('-chkpt', default='./chkpt')
    parser.add_argument('-trained_model', default='./chkpt/280_0.946.chkpt')
    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-is_shuffle', type=bool, default=True)
    parser.add_argument('-log', default='./log')
    parser.add_argument('-save_mode', default='best_and_interval')
    parser.add_argument('-save_interval', type=int, default=20)

    # network parameters
    parser.add_argument('-rnn_type', default='LSTM')
    parser.add_argument('-hidden', type=int, default=200)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-bidirectional', type=bool, default=True)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-wd', type=float, default=0.00001)
    parser.add_argument('-epoch', type=int, default=500)
    
    # loss parameters
    parser.add_argument('-alpha', type=float, default=0.0)
    parser.add_argument('-beta', type=float, default=1.0)
    parser.add_argument('-max_grad_norm', type=float, default=2.0)

    opt = parser.parse_args()
    print(opt)

    # device, here we use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load dataset
    data = torch.load(opt.data)
    train_data, valid_data = prepare_dataloaders(data, opt)

    if os.path.exists(opt.trained_model):
        print('[INFO] Continue train from checkpoint from: {}'.format(opt.trained_model))
        model = torch.load(opt.trained_model)
        state = model['model']
        setting = model['setting']
        start_i = model['epoch'] + 1
        # prepare model
        model = Seq2Seq(hidden=setting.hidden, rnn_type=opt.rnn_type,
                        bidirectional=setting.bidirectional, 
                        n_layers=setting.n_layers, dropout=setting.dropout,
                        n_pre_motions=PRE_MOTIONS, pre_trained_embedding=data['emb_table'], 
                        trg_dim=data['estimator'].n_components).to(device)
        # load trained state
        model.load_state_dict(state)
    else:
        # prepare model
        print('[INFO] Preparing seq2seq model.')
        model = Seq2Seq(hidden=opt.hidden, rnn_type=opt.rnn_type,
                        bidirectional=opt.bidirectional, 
                        n_layers=opt.n_layers, dropout=opt.dropout,
                        n_pre_motions=PRE_MOTIONS, pre_trained_embedding=data['emb_table'], 
                        trg_dim=data['estimator'].n_components).to(device)
        start_i = 0 # initial epoch
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    # train process
    train(model, train_data, valid_data, optimizer, device, opt, start_i)


if __name__ == '__main__':
    main()