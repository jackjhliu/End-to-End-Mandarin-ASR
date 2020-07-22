""" Compute error rate.
"""
import torch
import os
import argparse
import eval_utils


def main():
    parser = argparse.ArgumentParser(description="Compute error rate.")
    parser.add_argument('ckpt', type=str, help="Checkpoint to restore.")
    parser.add_argument('--split', default='test', type=str, help="Specify which split of data to evaluate.")
    parser.add_argument('--gpu_id', default=0, type=int, help="CUDA visible GPU ID. Currently only support single GPU.")
    parser.add_argument('--beams', default=1, type=int, help="Beam Search width.")
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
    if args.beams == 1:
        batch_size = cfg['train']['batch_size']
    else:
        batch_size = 1
    loader = data.load(split=args.split, batch_size=batch_size)

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
    error = eval_utils.get_error(loader, model, args.beams)
    print ("Error rate on %s set = %.4f" % (args.split, error))


if __name__ == '__main__':
    main()
