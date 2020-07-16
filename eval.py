""" Compute character error rate (CER).
"""
import torch
import os
import argparse
import eval_utils


def main():
    parser = argparse.ArgumentParser(description="Compute character error rate (CER).")
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
    print ("Dev CER of checkpoint: %.4f @epoch: %d" % (info['dev_cer'], info['epoch']))

    cfg = info['cfg']

    # Create dataset
    loader = data.load(split=args.split, batch_size=cfg['train']['batch_size'])

    # Build model
    tokenizer = torch.load('tokenizer.pth')
    model = build_model.Seq2Seq(len(tokenizer.vocab),
                                hidden_size=cfg['model']['hidden_size'],
                                encoder_layers=cfg['model']['encoder_layers'],
                                decoder_layers=cfg['model']['decoder_layers'])
    model.load_state_dict(info['weights'])
    model.eval()
    model = model.cuda()

    # Evaluate
    cer = eval_utils.get_cer(loader, model)
    print ("CER on %s set = %.4f" % (args.split, cer))


if __name__ == '__main__':
    main()