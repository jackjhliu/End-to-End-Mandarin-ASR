""" Inference on random input sentence and visualize the attention matrix.
"""
import torch
import os
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import argparse


def showAttention(output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    ax.set_yticklabels([''] + output_words)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def decode(s, vocab):
    s_out = []
    for i in s:
        w = vocab[i]
        if w == '<s>':
            continue
        elif w == '</s>':
            s_out.append(w)
            break
        s_out.append(w)
    return s_out


def main():
    parser = argparse.ArgumentParser(description="Inference on arbitrary input sentence and visualize the attention matrix.")
    parser.add_argument('ckpt', type=str, help="Checkpoint to restore.")
    parser.add_argument('--split', default='test', type=str, help="Specify which split of data to evaluate.")
    parser.add_argument('--gpu_id', default=0, type=int, help="CUDA visible GPU ID. Currently only support single GPU.")
    parser.add_argument('--root', default="./data_aishell", type=str, help="Directory of dataset.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    assert torch.cuda.is_available()
    import data
    import build_model

    # Restore checkpoint
    info = torch.load(args.ckpt)
    print ("Dev CER of checkpoint: %.4f @epoch: %d" % (info['dev_cer'], info['epoch']))

    cfg = info['cfg']

    # Create dataset
    loader, tokenizer = data.prepareData(root=args.root, split=args.split, batch_size=1)

    # Build model
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
            predictions = decode(predictions, tokenizer.vocab)
            attentions = attentions[:len(predictions)].cpu().numpy()   # (target_length, source_length)
            print ("Predict:")
            print (' '.join(predictions[:-1]))
            print ("Ground-truth:")
            print (tokenizer.decode(y[0,1:-1]))
            print ()
            showAttention(predictions, attentions)


if __name__ == '__main__':
    main()
