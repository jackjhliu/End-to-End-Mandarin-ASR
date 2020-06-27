""" Extract FBANK features from audio signals and compute necessary information for training. 
"""
import torch
import torchaudio
import os
import argparse
import glob
import data_utils
import pickle
from tqdm import tqdm
from torchnlp.encoders.text import StaticTokenizerEncoder


def compute_fbanks(root, split):
    """
    Args:
        root (string): The root directory of AISHELL dataset.
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
    """
    audio_files = glob.glob(os.path.join(root, "wav/%s/*/*.wav" % split))
    # Ignore audios without transcript.
    transcripts = data_utils.read_transcripts(root)
    audio_files = [a for a in audio_files if data_utils.get_id(a) in transcripts]

    for f in tqdm(audio_files):
        id = data_utils.get_id(f)
        out_fname = os.path.join(root, 'fbank', split, id+'.pth')
        if not os.path.exists(out_fname):
            x, _ = torchaudio.load(f)
            x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80)   # [n_windows, 80]
            torch.save(x, out_fname)


def compute_statistics(root):
    """
    Calculate the sequence lengths, mean, and standard deviation over the training set. During training
    the data generator will pool together examples with similar FBANK feature size, so we needs to
    calculate the sizes in advance.

    Args:
        root (string): The root directory of AISHELL dataset.
    """
    fbanks_train = glob.glob(os.path.join(root, "fbank/train/*.pth"))
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


def make_tokenizer(root):
    """
    Construct Pytorch-NLP sentence tokenizer for labellings.

    Args:
        root (string): The root directory of AISHELL dataset.
    """
    transcripts = data_utils.read_transcripts(root)
    fbanks_train = glob.glob(os.path.join(root, "fbank/train/*.pth"))
    labellings = [transcripts[data_utils.get_id(p)] for p in fbanks_train]
    tokenizer = StaticTokenizerEncoder(labellings,
                                       tokenize=data_utils.tokenize_fn,
                                       min_occurrences=5,
                                       append_eos=True,
                                       reserved_tokens=['<pad>', '<unk>', '</s>'])
    torch.save(tokenizer, os.path.join(root, 'tokenizer.pth'))


def main():
    parser = argparse.ArgumentParser(
        description="Extract FBANK features from audio signals and compute necessary information for training.")
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
        compute_statistics(args.root)

    print ("Make tokenizer ...")
    if not os.path.exists(os.path.join(args.root, 'tokenizer.pth')):
        make_tokenizer(args.root)

    print ("Completed !")


if __name__ == '__main__':
    main()

