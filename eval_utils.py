""" Define useful functions for evaluation.
"""
import torch
import editdistance


def decode(s, vocab):
    """
    Decode a label sequence.

    Args:
        s (list(integer)): A sentence represented by word indexes.
        vocab (dict): ID-to-word vocabulary. 

    Returns:
        s_out (list(string)): Decoded sentence.
    """
    s_out = []
    for i in s:
        w = vocab[i]
        if w == '<s>':
            continue
        elif w == '</s>':
            break
        else:
            s_out.append(w)
    return s_out


def get_cer(dataloader, model, vocab):
    """
    Calculate character error rate (CER) on a dataset.

    Args:
        vocab (dict): ID-to-word vocabulary. 
    """
    n_tokens = 0
    total_error = 0
    with torch.no_grad():
        for i, (xs, xlens, ys) in enumerate(dataloader):
            preds_batch, _ = model(xs.cuda(), xlens)   # [batch_size, 100]
            for j in range(preds_batch.shape[0]):
                preds = decode(preds_batch[j], vocab)
                gt = decode(ys[j], vocab)
                total_error += editdistance.eval(gt, preds)
                n_tokens += len(gt)
            print ("Calculating CER ... (#batch: %d/%d)" % (i+1, len(dataloader)), end='\r')
    print ()
    cer = total_error / n_tokens
    return cer

