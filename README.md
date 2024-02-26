# SingFake: Singing Voice Deepfake Detection

[![arXiv](https://img.shields.io/badge/arXiv-2309.07525-b31b1b.svg)](https://arxiv.org/abs/2309.07525)

 Official Repository for paper "SingFake: Singing Voice Deepfake Detection", in submission to ICASSP 2024. [[Project Webpage](https://singfake.org/)]

This repository is under construction. We are actively preparing files for ease of using.

## Updates
- Nov 2023: We released our metadata annotation tool (`annotation-tool/`) designed as a Chrome Extension to speed up the annotation process.
- Sep 2023: We released our training and evaluation scripts, ~~as well as trained model checkpoints for reproducibility~~ due to copyright concerns, we do not release our trained model checkpoints.

## Directory Structure
`dataset/` contains scripts related to preparing the dataset. Assuming you have a directory filled with downloaded FLAC files, you could run them first through `separate.py` to generate separated vocal stems and generate VAD timecodes, then use `split.py` to generate separated audio clips for training. We also provide `simulate_codec.py`, which is being used for generating our T03 subset.

`models/` contains script for training and evaluation of our four baseline models. `feat_resnet` contains implementation for Spectrogram+ResNet; `lfcc_resnet` contains implementation for LFCC+ResNet; `AASIST` and `wav2vec2+AASIST` contains their corresponding implementations.

## Annotation Tool
The metadata annotation tool is designed as a Chrome Extension built on top of a Google Firestore backend. To use this, you can start a free-tier project under Google Firebase, enable Firestore, and fill in your credentials under `annotation-tool/background.js`'s `firebaseConfig` variable.