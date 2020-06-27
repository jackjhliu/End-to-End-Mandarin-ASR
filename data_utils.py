""" Define useful functions for data I/O.
"""
import os
import glob


def tokenize_fn(s_in):
    s_out = ['<s>'] + list(s_in)
    return s_out


def get_id(audio_file):
    """
    Given an audio/fbank file path, return its ID.
    """
    return os.path.basename(audio_file)[:-4]


def read_transcripts(root):
    """
    Returns:
        transcripts (dict): All the transcripts for AISHELL dataset. They are represented
                            by {audio id: transcript}.
    """
    with open(os.path.join(root, "transcript/aishell_transcript_v0.8.txt")) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    transcripts = {}
    for l in lines:
        l = l.split()
        transcripts[l[0]] = ''.join(l[1:])
    return transcripts