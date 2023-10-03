# [Generalizing Voice Presentation Attack Detection to Unseen Synthetic Attacks and Channel Variation](https://link.springer.com/chapter/10.1007/978-981-19-5288-3_15)

## Abstract
Automatic Speaker Verification (ASV) systems aim to verify a speaker’s claimed identity through voice. However, voice can be easily forged with replay, text-to-speech (TTS), and voice conversion (VC) techniques, which may compromise ASV systems. Voice presentation attack detection (PAD) is developed to improve the reliability of speaker verification systems against such spoofing attacks. 
One main issue of voice PAD systems is its generalization ability to unseen synthetic attacks, i.e., synthesis methods that are not seen during training of the presentation attack detection models. We propose one-class learning, where the model compacts the distribution of learned representations of bona fide speech while pushing away spoofing attacks to improve the results. 
Another issue is the robustness to variations of acoustic and telecommunication channels. To alleviate this issue, we propose channel-robust training strategies, including data augmentation, multi-task learning, and adversarial learning. In this chapter, we analyze the two issues within the scope of synthetic attacks, i.e., TTS and VC, and demonstrate the effectiveness of our proposed methods. 

## To run our code
### Generalize to Unseen Synthetic Attacks with One-Class Learning
For our experiments in Section 2.2, we use the ResNet as the backbone and compare four one-class loss functions with the binary cross-entropy loss.
All the losses are available in the `-l` options: `isolate`, `scl`, `ocsoftmax`, `angulariso` (one-class losses as our narrative orders), 
`softmax`, `amsoftmax` (binary losses). Remember to specify where you want to save the model by `-o`.

An example of training and testing commands are as follows:
```
python3 train.py -o /data3/neil/hbas/models1101/resnet_ocsoftmax -l ocsoftmax --gpu 0 -m resnet
```

```
python3 test.py -t ASVspoof2019LA -m /data3/neil/hbas/models1101/resnet_ocsoftmax -l ocsoftmax --gpu 0
```

### Generalize to Channel Variation with Channel-Robust Strategies
For our experiments in Section 3.2 Table 2, we first compare different channel-robust training strategies. Simply add one of the following options to your command: `--AUG`, `--MT_AUG`, `--ADV_AUG`.
Remember to use `ASVspoof2019LASim` during testing by specifying `-t`.

An example of training and testing commands are as follows:
```
python3 train.py -o /data3/neil/hbas/models1101/rawnet -l ocsoftmax --gpu 0 -m rawnet --AUG --feat Raw --feat_len 80000
```

```
python3 test.py -t ASVspoof2019LASim -m /data3/neil/hbas/models1101/rawnet -l ocsoftmax --gpu 0
```

For our experiments in Section 3.2 Table 3, we then compare different augmentation and training strategies on the ASVspoof2021LA. Just set
whether you would like to use device augmentation or transmission augmentation or both.
Remember to use `ASVspoof2021LA` during testing by specifying `-t`.

An example of training and testing commands are as follows:
```
python3 train.py -o /data3/neil/hbas/models1101/resnet -l ocsoftmax --gpu 0 -m resnet --AUG --device_aug yes --transm_aug yes
```

```
python3 test.py -t ASVspoof2021LA -m /data3/neil/hbas/models1101/resnet -l ocsoftmax --gpu 0
```

### Citation
```
@Inbook{zhang2023generalizing,
author="Zhang, You
and Jiang, Fei
and Zhu, Ge
and Chen, Xinhui
and Duan, Zhiyao",
editor="Marcel, S{\'e}bastien
and Fierrez, Julian
and Evans, Nicholas",
title="Generalizing Voice Presentation Attack Detection to Unseen Synthetic Attacks and Channel Variation",
bookTitle="Handbook of Biometric Anti-Spoofing: Presentation Attack Detection and Vulnerability Assessment",
year="2023",
publisher="Springer Nature Singapore",
address="Singapore",
pages="421--443",
isbn="978-981-19-5288-3",
doi="10.1007/978-981-19-5288-3_15",
url="https://doi.org/10.1007/978-981-19-5288-3_15"
}
```

