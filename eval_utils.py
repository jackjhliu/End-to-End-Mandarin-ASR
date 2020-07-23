""" Define necessary functions for evaluation.
"""
import torch
import editdistance
from tqdm import tqdm


def get_error(dataloader, model, beam_width=1):
    """
    Calculate error rate on a specific dataset.
    """
    tokenizer = torch.load('tokenizer.pth')
    n_tokens = 0
    total_error = 0
    with torch.no_grad():
        eval_tqdm = tqdm(dataloader)
        for (xs, xlens, ys) in eval_tqdm:
            preds_batch, _ = model(xs.cuda(), xlens, beam_width=beam_width)   # [batch_size, 100]
            for i in range(preds_batch.shape[0]):
                preds = tokenizer.decode(preds_batch[i])
                gt = tokenizer.decode(ys[i])
                preds = preds.split()
                gt = gt.split()
                total_error += editdistance.eval(gt, preds)
                n_tokens += len(gt)
            eval_tqdm.set_description("Calculating error rate")
    error = total_error / n_tokens
    return error
