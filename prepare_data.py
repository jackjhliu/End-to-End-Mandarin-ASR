""" Extract FBANK features from audio signal and compute necessary statistics for training. 
"""
import torch
import torchaudio
import os
import argparse
import glob
import data_utils
from tqdm import tqdm


def compute_fbanks(root, split):
    """
    Args:
        root (string): The root directory of AISHELL dataset.
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
    """
    data_valid = data_utils.parse_partition(root, split)
    audio_files = glob.glob(os.path.join(root, "wav/%s/*/*.wav" % split))
    audio_files = [p for p in audio_files if data_utils.get_id(p) in data_valid]

    for f in tqdm(audio_files):
        id = data_utils.get_id(f)
        out_fname = os.path.join(root, 'fbank', split, id+'.pth')
        if not os.path.exists(out_fname):
            x, _ = torchaudio.load(f)
            x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80)   # [n_windows, 80]
            torch.save(x, out_fname)


def compute_statistics(root, fbanks_train):
    """
    Compute the sequence lengths, mean, and standard deviation over the training set. During training the data
    generator will pool together examples with similar FBANK feature size, so we needs to calculate the sizes
    in advance.

    Args:
        root (string): The root directory of AISHELL dataset.
        fbanks_train (list(string)): All of the FBANK feature file paths of training set.
    """
    xlens = {}
    # mean
    accumulate = 0
    n = 0
    print ("Calculating mean ...")
    for path in tqdm(fbanks_train):
        x = torch.load(path)
        assert x.shape[0] > 0
        id = data_utils.get_id(path)
        xlens[id] = x.shape[0]
        accumulate += x.sum(dim=0)
        n += x.shape[0]
    mean = accumulate / n
    # std
    accumulate = 0
    n = 0
    print ("Calculating std ...")
    for path in tqdm(fbanks_train):
        x = torch.load(path)
        accumulate += ((x-mean)**2).sum(dim=0)
        n += x.shape[0]
    std = torch.sqrt(accumulate / n)
    statistics = {'mean':mean, 'std': std, 'xlens': xlens}
    torch.save(statistics, os.path.join(root, 'statistics.pth'))


def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('root', type=str, help="The root directory of AISHELL dataset.")
    args = parser.parse_args()

    fbank_folder = os.path.join(args.root, 'fbank')
    if not os.path.exists(fbank_folder):
        os.mkdir(fbank_folder)
        os.mkdir(os.path.join(fbank_folder, 'train'))
        os.mkdir(os.path.join(fbank_folder, 'dev'))
        os.mkdir(os.path.join(fbank_folder, 'test'))

    print ("Computing FBANK features ...")
    compute_fbanks(args.root, 'train')
    compute_fbanks(args.root, 'dev')
    compute_fbanks(args.root, 'test')

    print ("Computing statistics ...")
    if not os.path.exists(os.path.join(args.root, 'statistics.pth')):
        fbanks_train = glob.glob(os.path.join(args.root, "fbank/train/*.pth"))
        compute_statistics(args.root, fbanks_train)

    print ("Completed !")


if __name__ == '__main__':
    main()

