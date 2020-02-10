import pickle
import argparse
import torch

from sklearn.decomposition import PCA


def load_processed_data(dataset_path, data_size):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
        # load dataset
        if data_size != -1:
            eye_dataset = dataset['eye_dataset'][:data_size]
        else:
            eye_dataset = dataset['eye_dataset'][:]
        # load learned estimator
        estimator = dataset['estimator']

    return eye_dataset, estimator


def get_data_pair(dataset):
    x = []
    y = []
    for data in dataset:
        for clip_info in data['clip_info']:
            for sents, landmarks in zip(clip_info['sent'], clip_info['landmarks']):
                x.append(sents[2])
                y.append(landmarks)

    print('[INFO] Dataset description.')
    print('\tPairs: {}'.format(len(x)))
    print('\tMax seq len in x:{}'.format(len(max(x, key=len))))
    print('\tMin seq len in x:{}'.format(len(min(x, key=len))))
    print('\tMax seq len in y:{}'.format(len(max(y, key=len))))
    print('\tMin seq len in y:{}'.format(len(min(y, key=len))))

    return x, y



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='./data/processed_eye_motion_dataset.pickle')
    parser.add_argument('-data_size', type=int, default=3)
    parser.add_argument('-emb', default='./data/glove.6B.300d.txt')
    parser.add_argument('-processed_path', default='./processed')

    parser.add_argument('-pca_components', type=int, default=10)
    opt = parser.parse_args()

    eye_dataset, estimator = load_processed_data(opt.dataset, opt.data_size)
    src_insts, trg_insts = get_data_pair(eye_dataset)

    # preprocessed_data = {
    #     'settings': opt,
    #     'pca': pca
    # }
    # print('[INFO] Dumping the processed data to pickle file: {}'.format(opt.processed_path))
    # torch.save(preprocessed_data, '{}/processed.pickle'.format(opt.processed_path))


if __name__ == '__main__':
    main()