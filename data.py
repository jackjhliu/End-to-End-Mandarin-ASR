""" Load and preprocess data.
"""
import torch
import torchaudio
import os
import pandas as pd
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader


class ASR(Dataset):
    """
    Stores a Pandas DataFrame in __init__, and reads and preprocesses examples in __getitem__.
    """
    def __init__(self, split):
        self.df = pd.read_csv('%s.csv' % split.upper())
        self.tokenizer = torch.load('tokenizer.pth')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            x (torch.FloatTensor, [seq_length, dim_features]): The FBANK features.
            y (torch.LongTensor, [n_tokens]): The phoneme sequence.
        """
        x, y = self.df.iloc[idx]
        x, _ = torchaudio.load(x)
        # Compute filter bank features
        x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80)   # [n_windows, 80]
        # Stack every 3 frames and down-sample frame rate by 3, following https://arxiv.org/pdf/1712.01769.pdf.
        x = x[:(x.shape[0]//3)*3].view(-1,3*80)   # [n_windows, 80] --> [n_windows//3, 240]
        # Tokenization
        y = self.tokenizer.encode(y)
        return x, y

    def generateBatch(self, batch):
        """
        Generate a mini-batch of data. For DataLoader's 'collate_fn'.

        Args:
            batch (list(tuple)): A mini-batch of (FBANK features, phoneme sequences) pairs.

        Returns:
            xs (torch.FloatTensor, [batch_size, (padded) seq_length, dim_features]): A mini-batch of FBANK features.
            xlens (torch.LongTensor, [batch_size]): Sequence lengths before padding.
            ys (torch.LongTensor, [batch_size, (padded) n_tokens]): A mini-batch of phoneme sequences.
        """
        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = rnn_utils.pad_sequence(xs, batch_first=True)   # [batch_size, (padded) seq_length, dim_features]
        ys = rnn_utils.pad_sequence(ys, batch_first=True)   # [batch_size, (padded) n_tokens]
        return xs, xlens, ys


def load(split, batch_size, workers=0):
    """
    Args:
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
        batch_size (integer): Batch size.
        workers (integer): How many subprocesses to use for data loading.

    Returns:
        loader (DataLoader): A DataLoader can generate batches of (FBANK features, FBANK lengths, phoneme sequence).
    """
    assert split in ['train', 'dev', 'test']

    dataset = ASR(split)
    print ("%s set size:"%split.upper(), len(dataset))
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=dataset.generateBatch,
                        shuffle=True,
                        num_workers=workers,
                        pin_memory=True)
    return loader


def inspect_data():
    """
    Test the functionality of input pipeline and visualize a few samples.
    """
    import matplotlib.pyplot as plt

    BATCH_SIZE = 64
    SPLIT = 'dev'

    loader = load(SPLIT, BATCH_SIZE)
    tokenizer = torch.load('tokenizer.pth')
    print ("Vocabulary size:", len(tokenizer.vocab))
    print (tokenizer.vocab)

    xs, xlens, ys = next(iter(loader))
    print (xs.shape, ys.shape)
    for i in range(BATCH_SIZE):
        print (ys[i])
        print (tokenizer.decode(ys[i]))
        plt.figure()
        plt.imshow(xs[i].T)
        plt.show()


if __name__ == '__main__':
    inspect_data()