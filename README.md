# End-to-End-Mandarin-ASR

中文語音辨識

End-to-end speech recognition on AISHELL dataset using Pytorch.

BiGRU encoder + Attention decoder, based on **"Listen, Attend and Spell"**<sup>[1](#References)</sup>.

The acoustic features are 80-dimensional filter banks. They are stacked every 3 consecutive frames, so the time resolution is reduced.

With this code you can achieve **~13% CER** on the test set.

## Requirements
* Python 3.6
* Pytorch 1.5
* torchaudio 0.5.0
* [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) 0.5.0
* PyYAML
* editdistance
* matplotlib
* tqdm

## Usage
### Data
1. Download AISHELL dataset (data_aishell.tgz) from http://www.openslr.org/33/.
2. Extract data_aishell.tgz:
```bash
$ python extract_aishell.py ${PATH_TO_data_aishell.tgz}
```
3. Extract filter bank features and prepare for data normalization:
```bash
$ python prepare_data.py ${PATH_TO_AISHELL}
```

### Train
Check available options:
```bash
$ python train.py -h
```
Use the default configuration for training:
```bash
$ python train.py exp/default.yaml
```
You can also write your own configuration file based on `exp/default.yaml`.
```bash
$ python train.py ${PATH_TO_YOUR_CONFIG}
```

### Show loss curve
With the default configuration, the training logs are stored in `exp/default/history.txt`.
You should specify your training logs accordingly.
```bash
$ python vis_hist.py exp/default/history.txt
```
![](./img/Figure_1.png)

### Test
During training, the program will keep monitoring the error rate on development set.
The checkpoint with the lowest error rate will be saved in the logging directory (by default `exp/default/best.pth`).

To evalutate the checkpoint on test set, run:
```bash
$ python eval.py exp/default/best.pth
```

Or you can test random audio from the test set and see the attentions:
```bash
$ python vis_attns.py exp/default/best.pth

Predict:
北 国 将 不 再 生 产 大 压 无 人 机
Ground-truth:
美 国 将 不 再 生 产 大 鸦 无 人 机
```
![](./img/Figure_3.png)

## TODO
- [ ] Beam Search
- [ ] Restore checkpoint and resume previous training
- [ ] LM Rescoring
- [ ] Label Smoothing
- [ ] Polyak Averaging

## References
[1] W. Chan _et al._, "Listen, Attend and Spell",
https://arxiv.org/pdf/1508.01211.pdf

[2] J. Chorowski _et al._, "Attention-Based Models for Speech Recognition",
https://arxiv.org/pdf/1506.07503.pdf

[3] M. Luong _et al._, "Effective Approaches to Attention-based Neural Machine Translation",
https://arxiv.org/pdf/1508.04025.pdf