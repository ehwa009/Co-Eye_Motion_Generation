

def collate_fn(insts, opt):
    seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, trg_insts = list(zip(*seq_pairs))
    max_src_len = max(len(inst) for inst in src_insts)
    max_trg_len = max(len(inst) for inst in trg_insts)
    
    frame_duration = opt.frame_duration
    sp_duration = max_src_len / opt.speech_sp
    pre_duration = opt.pre_motions * frame_duration
    expression_duration = opt.estimation_expression * frame_duration

    num_words_for_pre_motion = round(max_src_len * pre_duration / sp_duration)
    num_words_for_estimation = 