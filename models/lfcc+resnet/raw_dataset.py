#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import scipy.io as sio
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate
import warnings


def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=16000)
    wave = librosa.util.normalize(wave)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

class ASVspoof2019Raw(Dataset):
    def __init__(self, access_type, path_to_database, path_to_protocol, part='train'):
        super(ASVspoof2019Raw, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        if self.part == "eval":
            protocol = os.path.join(self.ptd, access_type, 'ASVspoof2019_' + access_type +
                                    '_cm_protocols/ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename + ".flac")
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)


class VCC2020Raw(Dataset):
    def __init__(self, path_to_spoof="/data2/neil/nii-yamagishilab-VCC2020-listeningtest-31f913c", path_to_bonafide="/data2/neil/nii-yamagishilab-VCC2020-database-0b2fb2e"):
        super(VCC2020Raw, self).__init__()
        self.all_spoof = librosa.util.find_files(path_to_spoof, ext="wav")
        self.all_bonafide = librosa.util.find_files(path_to_bonafide, ext="wav")

    def __len__(self):
        # print(len(self.all_spoof), len(self.all_bonafide))
        return len(self.all_spoof) + len(self.all_bonafide)

    def __getitem__(self, idx):
        if idx < len(self.all_bonafide):
            filepath = self.all_bonafide[idx]
            label = "bonafide"
            filename = "_".join(filepath.split("/")[-3:])[:-4]
            tag = "-"
        else:
            filepath = self.all_spoof[idx - len(self.all_bonafide)]
            filename = os.path.basename(filepath)[:-4]
            label = "spoof"
            tag = filepath.split("/")[-3]
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2015Raw(Dataset):
    def __init__(self, path_to_database="/data/neil/ASVspoof2015/wav", path_to_protocol="/data/neil/ASVspoof2015/CM_protocol", part='train'):
        super(ASVspoof2015Raw, self).__init__()
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, self.part)
        self.path_to_protocol = path_to_protocol
        cm_pro_dict = {"train": "cm_train.trn", "dev": "cm_develop.ndx", "eval": "cm_evaluation.ndx"}
        protocol = os.path.join(self.path_to_protocol, cm_pro_dict[self.part])
        self.tag = {"human": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5,
                    "S6": 6, "S7": 7, "S8": 8, "S9": 9, "S10": 10}
        self.label = {"spoof": 1, "human": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, tag, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, speaker, filename + ".wav")
        waveform, sr = torchaudio_load(filepath)
        filename = filename.replace("_", "-")
        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2019LARaw_withChannel(Dataset):
    def __init__(self, access_type="LA", path_to_database="/data/shared/ASVspoof2019Channel", path_to_protocol="/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part='train'):
        super(ASVspoof2019LARaw_withChannel, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = path_to_database
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(self.path_to_protocol,
                                'ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        if self.part == "eval":
            protocol = os.path.join(self.ptd, access_type, 'ASVspoof2019_' + access_type +
                                    '_cm_protocols/ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8,
                    "A09": 9,
                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17,
                    "A18": 18,
                    "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.channel = ['amr[br=5k15]', 'amrwb[br=15k85]', 'g711[law=u]', 'g722[br=56k]',
                        'g722[br=64k]', 'g726[law=a,br=16k]', 'g728', 'g729a', 'gsmfr',
                        'silk[br=20k]', 'silk[br=5k]', 'silkwb[br=10k,loss=5]', 'silkwb[br=30k]']

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info) * len(self.channel)

    def __getitem__(self, idx):
        file_idx = idx // len(self.channel)
        channel_idx = idx % len(self.channel)
        speaker, filename, _, tag, label = self.all_info[file_idx]
        channel = self.channel[channel_idx]
        filepath = os.path.join(self.path_to_audio, filename + "_" + channel + ".wav")
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label, channel

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2019LARaw_withDevice(Dataset):
    def __init__(self, access_type="LA", path_to_database="/data/shared/ASVspoof2019LA-Sim", path_to_protocol="/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/", part='eval'):
        super(ASVspoof2019LARaw_withDevice, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = path_to_database
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(self.path_to_protocol,
                                'ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8,
                    "A09": 9,
                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17,
                    "A18": 18,
                    "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.devices = ['AKSPKRS80sUk002-16000', 'AKSPKRSVinUk002-16000', 'Doremi-16000', 'RCAPB90-16000',
                        'ResloRBRedLabel-16000', 'AKSPKRSSpeaker002-16000', 'BehritoneirRecording-16000',
                        'OktavaML19-16000', 'ResloRB250-16000', 'SonyC37Fet-16000', 'iPadirRecording-16000', 'iPhoneirRecording-16000']

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info) * len(self.devices)

    def __getitem__(self, idx):
        file_idx = idx // len(self.devices)
        device_idx = idx % len(self.devices)
        speaker, filename, _, tag, label = self.all_info[file_idx]
        device = self.devices[device_idx]
        filepath = os.path.join(self.path_to_audio, device, filename + ".wav")
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label, device

    def collate_fn(self, samples):
        return default_collate(samples)

if __name__ == "__main__":

    asvspoof2019channel = ASVspoof2019LARaw_withChannel()
    print(len(asvspoof2019channel))
    waveform, filename, tag, label, channel = asvspoof2019channel[123]
    print(waveform.shape)
    print(filename)
    print(tag)
    print(label)
    print(channel)
    pass
