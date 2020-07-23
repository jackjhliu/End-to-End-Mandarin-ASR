""" Train the model.
"""
import yaml
import os
import re
import argparse
import time
import torch
import eval_utils
from tqdm import tqdm


def get_lr(optimizer):
    """
    A helper function to retrieve the solver's learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def log_history(save_path, message):
    """
    A helper function to log the history.
    The history text file is saved as: {SAVE_PATH}/history.txt

    Args:
        save_path (string): The location to log the history.
        message (string): The message to log.
    """
    fname = os.path.join(save_path,'history.csv')
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write("datetime,epoch,learning rate,train loss,dev loss,error rate\n")
            f.write("%s\n" % message)
    else:
        with open(fname, 'a') as f:
            f.write("%s\n" % message)


def save_checkpoint(filename, save_path, epoch, dev_error, cfg, model, optimizer, scheduler):
    filename = os.path.join(save_path, filename)
    info = {'epoch': epoch,
            'dev_error': dev_error,
            'cfg': cfg,
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()}
    torch.save(info, filename)


def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('cfg', type=str, help="Specify which experiment config file to use.")
    parser.add_argument('--gpu_id', default=0, type=int, help="CUDA visible GPU ID. Currently only support single GPU.")
    parser.add_argument('--workers', default=0, type=int, help="How many subprocesses to use for data loading.")
    parser.add_argument('--restore', type=str, help="Specify a checkpoint to restore weights from.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    assert torch.cuda.is_available()
    import data
    import build_model

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not cfg['logdir']:
        save_path = os.path.splitext(args.cfg)[0]
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Create dataset
    train_loader = data.load(split='train', batch_size=cfg['train']['batch_size'], workers=args.workers)
    dev_loader = data.load(split='dev', batch_size=cfg['train']['batch_size'], workers=args.workers)

    # Build model
    tokenizer = torch.load('tokenizer.pth')
    model = build_model.Seq2Seq(len(tokenizer.vocab),
                                hidden_size=cfg['model']['hidden_size'],
                                encoder_layers=cfg['model']['encoder_layers'],
                                decoder_layers=cfg['model']['decoder_layers'],
                                drop_p=cfg['model']['drop_p'])
    model = model.cuda()

    # Training criteria
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['init_lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=cfg['train']['decay_factor'],
                                                           patience=cfg['train']['patience'],
                                                           min_lr=1e-6)

    # Load checkpoints
    if args.restore:
        info = torch.load(args.restore)
        epoch = info['epoch']
        model.load_state_dict(info['weights'])
        optimizer.load_state_dict(info['optimizer'])
        scheduler.load_state_dict(info['scheduler'])
    else:
        epoch = 0

    if os.path.exists(os.path.join(save_path, 'best.pth')):
        info = torch.load(os.path.join(save_path, 'best.pth'))
        best_epoch = info['epoch']
        best_error = info['dev_error']
    else:
        best_epoch = 0
        best_error = float('inf')


    while (1):
        print ("---")
        epoch += 1
        print ("Epoch: %d" % (epoch))
        # Show learning rate
        lr = get_lr(optimizer)
        print ("Learning rate: %f" % lr)

        # Training loop
        model.train()
        train_loss = 0
        n_tokens = 0
        train_tqdm = tqdm(train_loader)
        for (xs, xlens, ys) in train_tqdm:
            loss = model(xs.cuda(), xlens, ys.cuda())
            train_loss += loss.item() * (ys[:,1:] > 0).long().sum()
            n_tokens += (ys[:,1:] > 0).long().sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)   # Gradient clipping
            optimizer.step()

            train_tqdm.set_description("Training")
            train_tqdm.set_postfix(loss="%.3f" % (train_loss / n_tokens))
        train_loss = train_loss / n_tokens

        # Validation loop
        model.eval()
        # Compute dev loss
        dev_loss = 0
        n_tokens = 0
        with torch.no_grad():
            for (xs, xlens, ys) in dev_loader:
                dev_loss += model(xs.cuda(), xlens, ys.cuda()).item() * (ys[:,1:] > 0).long().sum()
                n_tokens += (ys[:,1:] > 0).long().sum()
        dev_loss = dev_loss / n_tokens
        # Compute dev error rate
        error = eval_utils.get_error(dev_loader, model)
        print ("Dev. loss: %.3f," % dev_loss, end=' ')
        print ("dev. error rate: %.4f" % error)
        if error < best_error:
            best_error = error
            best_epoch = epoch
            # Save best model
            save_checkpoint("best.pth", save_path, best_epoch, best_error, cfg, model, optimizer, scheduler)
        print ("Best dev. error rate: %.4f @epoch: %d" % (best_error, best_epoch))

        scheduler.step(error)

        # Save checkpoint
        save_checkpoint("last.pth", save_path, epoch, error, cfg, model, optimizer, scheduler)

        # Logging
        datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        msg = "%s,%d,%f,%f,%f,%f" % (datetime, epoch, lr, train_loss, dev_loss, error)
        log_history(save_path, msg)


if __name__ == '__main__':
    main()
