""" Compute error rate.
"""
import torch
import os
import argparse
import eval_utils


def main():
    parser = argparse.ArgumentParser(description="Compute error rate.")
    parser.add_argument('LAS', type=str, help="LAS checkpoint.")
    parser.add_argument('--bs', default=64, type=int, help="Batch size.")
    parser.add_argument('--LM', type=str, help="Language model checkpoint.")
    parser.add_argument('--beams', default=1, type=int, help="Beam Search width.")
    parser.add_argument('--fusion', default=0.3, type=float, help="Language model shallow fusion factor.")
    parser.add_argument('--split', default='test', type=str, help="Specify which split of data to evaluate.")
    parser.add_argument('--gpu_id', default=0, type=int, help="CUDA visible GPU ID. Currently only support single GPU.")
    parser.add_argument('--workers', default=0, type=int, help="How many subprocesses to use for data loading.")
    args = parser.parse_args()

    assert not (args.bs > 1 and args.beams > 1), ("Only Greedy Search (beams=1) supports batch_size > 1.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    assert torch.cuda.is_available()
    import data
    import build_model

    # Load data
    loader = data.load(split=args.split, batch_size=args.bs, workers=args.workers)
    tokenizer = torch.load('tokenizer.pth')

    # LAS model
    info = torch.load(args.LAS)
    cfg = info['cfg']
    LAS = build_model.Seq2Seq(len(tokenizer.vocab),
                              hidden_size=cfg['model']['hidden_size'],
                              encoder_layers=cfg['model']['encoder_layers'],
                              decoder_layers=cfg['model']['decoder_layers'],
                              use_bn=cfg['model']['use_bn'])
    LAS.load_state_dict(info['weights'])
    LAS.eval()
    LAS = LAS.cuda()

    # Language model
    if args.LM:
        info = torch.load(args.LM)
        cfg = info['cfg']
        LM = build_model.LM(len(tokenizer.vocab),
                            hidden_size=cfg['model']['hidden_size'],
                            num_layers=cfg['model']['layers'])
        LM.load_state_dict(info['weights'])
        LM.eval()
        LM = LM.cuda()
    else:
        LM = None

    # Evaluate
    _, error = eval_utils.eval_dataset(loader, LAS, args.beams, LM, args.fusion)
    print ("Error rate on %s set = %.4f" % (args.split, error))


if __name__ == '__main__':
    main()
