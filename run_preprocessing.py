import pickle
import argparse
import torch

from sklearn.decomposition import PCA

def load_data(path, data_size):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if data_size < len(data):
        dataset = data[:data_size]
    else:
        dataset = data[:]

    return dataset

def get_data_pair(data):
    x = []
    y = []
    
    for d in data:
        for ci in d['clip_info']:
            for sents, landmarks in zip(ci['sent'], ci['landmarks']):
                x.append(sents[2])
                y.append(landmarks)

    print('[INFO] Dataset description.')
    print('\tPairs: {}'.format(len(x)))
    print('\tMax seq len in x:{}'.format(len(max(x, key=len))))
    print('\tMin seq len in x:{}'.format(len(min(x, key=len))))
    print('\tMax seq len in y:{}'.format(len(max(y, key=len))))
    print('\tMin seq len in y:{}'.format(len(min(y, key=len))))

    return x, y

def trg_insts_norm(trg_insts):
    tmp = []
    length = []
    for inst in trg_insts:
        inst_count = 0
        for landmark in inst:
            tmp.append(landmark)
            inst_count += 1
        length.append(inst_count)

    return tmp, length

def run_pca(trg_insts, lengths, n_components):
    pca = PCA(n_components=n_components)
    trg_pca = pca.fit_transform(trg_insts)

    return pca

def show_pca_subspace(pca):
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='./data/eye_motion_dataset.pickle')
    parser.add_argument('-data_size', type=int, default=10)
    parser.add_argument('-emb', default='./data/glove.6B.300d.txt')

    parser.add_argument('-pca_components', type=int, default=10)
    opt = parser.parse_args()

    eye_dataset = load_data(opt.dataset, opt.data_size)
    src_insts, trg_insts = get_data_pair(eye_dataset)

    # normalize trg_insts
    expaned_trg_insts, lenghts = trg_insts_norm(trg_insts)

    # run pca
    pca = run_pca(expaned_trg_insts, lenghts, opt.pca_components)

    preprocessed_data = {
        'settings': opt,
        'pca': pca
    }


    print('TEST')


if __name__ == '__main__':
    main()