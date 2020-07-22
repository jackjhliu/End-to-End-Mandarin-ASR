""" Define necessary functions for evaluation.
"""
import torch
import editdistance


def get_error(dataloader, model, beam_width=1):
    """
    Calculate error rate on a specific dataset.
    """
    tokenizer = torch.load('tokenizer.pth')
    n_tokens = 0
    total_error = 0
    with torch.no_grad():
        for i, (xs, xlens, ys) in enumerate(dataloader):
            preds_batch, _ = model(xs.cuda(), xlens, beam_width=beam_width)   # [batch_size, 100]
            for j in range(preds_batch.shape[0]):
                preds = tokenizer.decode(preds_batch[j])
                gt = tokenizer.decode(ys[j])
                preds = preds.split()
                gt = gt.split()
                total_error += editdistance.eval(gt, preds)
                n_tokens += len(gt)
            print ("Calculating error rate ... (#batch: %d/%d)" % (i+1, len(dataloader)), end='\r')
    print ()
    error = total_error / n_tokens
    return error
