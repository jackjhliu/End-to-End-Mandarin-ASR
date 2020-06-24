""" Define useful functions for data I/O.
"""
import os
import glob


def get_id(audio_file):
    """
    Given the audio/fbank file path, return its ID.
    """
    return os.path.basename(audio_file)[:-4]


def parse_partition(root, split):
    """
    Create pairs of {audio id: transcript} examples for a specified partition of data.

    Args:
        root (string): The root directory of AISHELL dataset.
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.

    Returns:
        data_valid (dict): All of the {audio id: transcript} pairs belong to this partition.
    """
    # Load_data
    with open(os.path.join(root, "transcript/aishell_transcript_v0.8.txt")) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    data = {}   # Representing {audio id: transcript}.
    for l in lines:
        l = l.split()
        data[l[0]] = ''.join(l[1:])

    # Filter by split
    split_list = glob.glob(os.path.join(root, "wav/%s/*/*.wav" % split))
    split_list = [get_id(p) for p in split_list]
    data_valid = {}
    for id in split_list:
        if id in data:
            data_valid[id] = data[id]
    return data_valid
