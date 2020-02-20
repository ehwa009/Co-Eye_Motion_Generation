import numpy as np

from constant import *


def collate_fn(insts, opt):
    seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, trg_insts = list(zip(*seq_pairs))
    # find max length of each seqeunce
    max_src_len = max(len(inst) for inst in src_insts)
    max_trg_len = max(len(inst) for inst in trg_insts)
    
    sp_duration = max_src_len / SPEECH_SPEED
    pre_duration = PRE_MOTIONS * FRAME_DURATION
    expression_duration = ESTIMATION_MOTIONS * FRAME_DURATION

    num_words_for_pre_motion = round(max_src_len * pre_duration / sp_duration)
    num_words_for_estimation = round(max_src_len * expression_duration / sp_duration)

    # padding
    padded_src = np.array([inst + [PAD] * (max_src_len - len(inst)) 
                                                for inst in src_insts])
    for inst in trg_insts:
        trg_pad = []
        if max_trg_len - len(inst) > 0:
            for i in range(max_trg_len - len(inst)):
                trg_pad.append([0] * 15)
            inst += trg_pad
    padded_trg = np.array(trg_insts)