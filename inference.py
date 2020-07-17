""" Test on random audio from dataset and visualize the attention matrix.
"""
import torch
import os
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import argparse


def showAttention(predictions, attentions):
    output_words = predictions.split()
    # Set up figure with colorbar
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    ax.set_yticklabels([''] + output_words)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test on random audio from dataset and visualize the attention matrix.")
    parser.add_argument('ckpt', type=str, help="Checkpoint to restore.")
    parser.add_argument('--split', default='test', type=str, help="Specify which split of data to evaluate.")
    parser.add_argument('--gpu_id', default=0, type=int, help="CUDA visible GPU ID. Currently only support single GPU.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    assert torch.cuda.is_available()
    import data
    import build_model

    # Restore checkpoint
    info = torch.load(args.ckpt)
    print ("Dev. error rate of checkpoint: %.4f @epoch: %d" % (info['dev_error'], info['epoch']))

    cfg = info['cfg']

    # Create dataset
    loader = data.load(split=args.split, batch_size=1)

    # Build model
    tokenizer = torch.load('tokenizer.pth')
    model = build_model.Seq2Seq(len(tokenizer.vocab),
                                hidden_size=cfg['model']['hidden_size'],
                                encoder_layers=cfg['model']['encoder_layers'],
                                decoder_layers=cfg['model']['decoder_layers'])
    model.load_state_dict(info['weights'])
    model.eval()
    model = model.cuda()

    # Inference
    with torch.no_grad():
        for (x, xlens, y) in loader:
            predictions, attentions = model(x.cuda(), xlens)
            predictions, attentions = predictions[0], attentions[0]
            predictions = tokenizer.decode(predictions)
            attentions = attentions[:len(predictions.split())].cpu().numpy()   # (target_length, source_length)
            ground_truth = tokenizer.decode(y[0])
            print ("Predict:")
            print (predictions)
            print ("Ground-truth:")
            print (ground_truth)
            print ()
            showAttention(predictions, attentions)


if __name__ == '__main__':
    main()