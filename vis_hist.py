""" Visualize training history.
"""
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse


def main():
    parser = argparse.ArgumentParser(description="Visualize training history.")
    parser.add_argument('history', type=str, help="Path to the history file.")
    args = parser.parse_args()

    lines = list(csv.reader(open(args.history)))
    lines = sorted(lines, key=lambda l:int(l[1]))
    # format: datetime, epoch, LR, train_loss, dev_loss, PER
    _, epochs, LRs, train_losses, dev_losses, PERs = zip(*lines)

    plt.figure(figsize=(15,3))
    plt.subplots_adjust(.05, 0.15, .95, .9, None, None)

    plt.subplot(1,3,1)
    plt.title("Loss")
    plt.plot(np.int64(epochs), np.float32(train_losses), label='train')
    plt.plot(np.int64(epochs), np.float32(dev_losses), label='dev')
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1,3,2)
    plt.title("Dev PER")
    plt.grid()
    plt.plot(np.int64(epochs), np.float32(PERs))
    plt.ylim(0,1)
    plt.xlabel('epochs')

    plt.subplot(1,3,3)
    plt.title("Learning rate")
    plt.plot(np.int64(epochs), np.float32(LRs))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('epochs')

    plt.show()


if __name__ == '__main__':
    main()
