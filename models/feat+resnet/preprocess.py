import raw_dataset as dataset
from audio_feature_extraction import LFCC
import os
import torch
from tqdm import tqdm
from torchaudio import transforms
from scipy.fftpack import fft, ifft, fftshift, ifftshift, next_fast_len
import numpy as np
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)
device = torch.device("cuda" if cuda else "cpu")


# ## LFCC
# for part_ in ["train", "dev", "eval"]:
#     asvspoof_raw = dataset.ASVspoof2019Raw("LA", "/data1/neil/DS_10283_3336/", "/data1/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part=part_)
#     target_dir = os.path.join("/data2/neil/ASVspoof2019LA", part_, "LFCC")
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#     lfcc = LFCC(320, 160, 512, 16000, 20, with_energy=False)
#     lfcc = lfcc.to(device)
#     for idx in tqdm(range(len(asvspoof_raw))):
#         # if idx > 0: break # debug purpose
#         waveform, filename, tag, label = asvspoof_raw[idx]
#         # waveform = spectral_whitening(waveform.squeeze(0).numpy())
#         # waveform = torch.from_numpy(np.expand_dims(waveform, axis=0))
#         waveform = waveform.to(device)
#         lfccOfWav = lfcc(waveform)
#         # print(lfccOfWav.shape)
#         torch.save(lfccOfWav, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
#     print("Done!")

## MFCC
for part_ in ["train", "dev", "eval"]:
    asvspoof_raw = dataset.ASVspoof2019Raw("LA", "/data1/neil/DS_10283_3336/", "/data1/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part=part_)
    target_dir = os.path.join("/data2/neil/ASVspoof2019LA", part_, "MFCC")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    transform = transforms.MFCC(sample_rate=16000, n_mfcc=60, melkwargs={"n_fft": 320, "hop_length": 160, "center": False})
    transform = transform.to(device)
    for idx in tqdm(range(len(asvspoof_raw))):
        # if idx > 0: break
        waveform, filename, tag, label = asvspoof_raw[idx]
        # waveform = spectral_whitening(waveform.squeeze(0).numpy())
        # waveform = torch.from_numpy(np.expand_dims(waveform, axis=0))
        waveform = waveform.to(device)
        mfccOfWav = transform(waveform).transpose(2, 1)
        # print(mfccOfWav.shape)
        torch.save(mfccOfWav, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
    print("Done!")
