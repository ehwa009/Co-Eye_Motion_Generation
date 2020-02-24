import argparse
import time
import torch
import numpy as np

from data_utils import normalize_string
from constant import *
from Seq2Eye.model import Seq2Seq
from collections import namedtuple
from display import Display


def inference(model, input_words, pre_motion_seq, word2idx):
    input_seq = np.zeros((len(input_words) + 2, 1)) # s x b
    input_seq[0] = SOS 
    input_seq[-1] = EOS
    for i, word in enumerate(input_words):
        try:
            word_idx = word2idx[word]
        except KeyError:
            word_idx = UNK
        input_seq[i + 1, 0] = word_idx
    
    # convert array to tensor
    input_seq = torch.LongTensor(input_seq)
    pre_motion_seq = torch.FloatTensor(pre_motion_seq)

    # encoder forward
    enc_out, enc_hid = model.encoder(input_seq)
    dec_hid = enc_hid[:model.decoder.n_layers]

    trg_len = PRE_MOTIONS + ESTIMATION_MOTIONS
    motion_out = np.array([])
    attns = torch.zeros((trg_len, len(input_seq)))

    for t in range(trg_len):
        if t < PRE_MOTIONS:
            dec_in = pre_motion_seq[t].unsqueeze(0)
            dec_out, dec_hid, attn_weight = model.decoder(dec_in, dec_hid, enc_out)
        else:
            dec_in = dec_out
            dec_out, dec_hid, attn_weight = model.decoder(dec_in, dec_hid, enc_out)
            dec_in = dec_out
            if t == PRE_MOTIONS:
                motion_out = dec_out.data.cpu().numpy()
            else:
                motion_out = np.vstack((motion_out, dec_out.data.cpu().numpy()))
            
        if attn_weight is not None:
            attns[t] = attn_weight.data

    return motion_out, attns


def infer_from_words(model, pca_n_components, words, word2idx):
    total_motion_frames = PRE_MOTIONS + ESTIMATION_MOTIONS
    start = time.time()
    sp_duration = len(words) / SPEECH_SPEED
    pre_duration = PRE_MOTIONS * FRAME_DURATION
    expression_duration = ESTIMATION_MOTIONS * FRAME_DURATION

    num_words_for_pre_motion = round(len(words) * pre_duration / sp_duration)
    num_words_for_estimation = round(len(words) * expression_duration / sp_duration)
    
    padded_words = ['UNK'] * num_words_for_pre_motion + words
    pre_motion_seq = np.zeros((total_motion_frames, pca_n_components))

    output_tuple = namedtuple('InferenceOutput', ['words', 'pre_motion_seq', 'out_motion', 'attention'])
    outputs = []
    for i in range(0, len(padded_words) - num_words_for_pre_motion, num_words_for_estimation):
        sample_words = padded_words[i:i + num_words_for_pre_motion + num_words_for_estimation]
        with torch.no_grad():
            output, attn = inference(model, sample_words, pre_motion_seq, word2idx)
        outputs.append(output_tuple(sample_words, pre_motion_seq, output, attn))
        pre_motion_seq = np.asarray(output)[:PRE_MOTIONS]

    return outputs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='./processed/processed_final.pickle')
    # parser.add_argument('-chkpt', default='./chkpt/eye_model.chkpt')
    parser.add_argument('-chkpt', default='./chkpt/300_0.469.chkpt')
    # parser.add_argument('-chkpt', default='./chkpt/lr_0.0001_batch_size_512/eye_model.chkpt')
    
    opt = parser.parse_args()

    # sent = 'Physical education usually involves studying human biology and exercise'
    sent = "it's really not a good thing to do no matter what stage of life you re in"

    # load data
    data = torch.load(opt.data)
    # load trained model
    trained_model = torch.load(opt.chkpt)
    state = trained_model['model']
    settings = trained_model['setting']
    
    # prepare model
    model = Seq2Seq(hidden=settings.hidden, bidirectional=settings.bidirectional, 
                    n_layers=settings.n_layers, dropout=settings.dropout,
                    pre_trained_embedding=data['emb_table'], 
                    trg_dim=data['estimator'].n_components)
    model.load_state_dict(state)
    model.eval()
    
    # normalize and split input sentence
    words = normalize_string(sent).split()
    outputs = infer_from_words(model, data['estimator'].n_components, 
                                words, data['lang'].word2index)

    # pca inverse transform
    eye_motion_list = []
    for output in outputs:
        for motion in output.out_motion:
            transformed = data['estimator'].inverse_transform(motion)
            transformed = [int(trans) for trans in transformed.tolist()]
            eye_motion_list.append(transformed)

    # display infered output
    display = Display(180, 320) # 320 x 180
    display.display(eye_motion_list)





if __name__ == '__main__':
    main()
    