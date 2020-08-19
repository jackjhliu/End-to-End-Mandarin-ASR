""" Define necessary functions for evaluation.
"""
import torch
import editdistance
import numpy as np
from tqdm import tqdm


def eval_dataset(dataloader, LAS, beam_width=1, LM=None, fusion=0.3):
    """
    Calculate loss and error rate on a dataset.

    Args:
        LAS (nn.Module): LAS model.
        LM (nn.Module): Language model for shallow fusion.
        fusion (float): Language model shallow fusion factor.
    """
    tokenizer = torch.load('tokenizer.pth')
    total_loss = []
    n_tokens = 0
    total_error = 0
    with torch.no_grad():
        eval_tqdm = tqdm(dataloader, desc="Evaluating")
        for (xs, xlens, ys) in eval_tqdm:
            total_loss.append(LAS(xs.cuda(), xlens, ys.cuda()).item())
            preds_batch, _ = LAS(xs.cuda(), xlens, beam_width=beam_width, LM=LM, fusion=fusion)
            for i in range(preds_batch.shape[0]):
                preds = tokenizer.decode(preds_batch[i])
                gt = tokenizer.decode(ys[i])
                preds = preds.split()
                gt = gt.split()
                total_error += editdistance.eval(gt, preds)
                n_tokens += len(gt)
            # Show message
            loss = np.mean(total_loss)
            error = total_error / n_tokens
            eval_tqdm.set_postfix(loss="%.3f"%loss, error="%.4f"%error)
    return loss, error
