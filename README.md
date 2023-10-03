# SingFake: Singing Voice Deepfake Detection
 Official Repository for paper "SingFake: Singing Voice Deepfake Detection", in submission to ICASSP 2024.

This repository is under construction. We are actively preparing files for ease of using.

## Directory Structure
`dataset/` contains scripts related to preparing the dataset. Assuming you have a directory filled with downloaded FLAC files, you could run them first through `separate.py` to generate separated vocal stems and generate VAD timecodes, then use `split.py` to generate separated audio clips for training. We also provide `simulate_codec.py`, which is being used for generating our T03 subset.

`models/` contains script for training and evaluation of our four baseline models. `feat_resnet` contains implementation for both Spectrogram+ResNet and LFCC+ResNet; `AASIST` and `wav2vec2+AASIST` contains their corresponding implementations.
