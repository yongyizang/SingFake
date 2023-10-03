import numpy as np
import soundfile as sf
import torch, os
from torch import Tensor
from torch.utils.data import Dataset
import librosa

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key

class Dataset_SingFake(Dataset):
    def __init__(self, base_dir, is_mixture=False, target_sr=16000):
        """
        base_dir should contain mixtures/ and vocals/ folders
        """
        self.base_dir = base_dir
        self.is_mixture = is_mixture
        self.target_sr = target_sr
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        
        # get file list
        self.file_list = []
        if self.is_mixture:
            self.target_path = os.path.join(self.base_dir, "mixtures")
        else:
            self.target_path = os.path.join(self.base_dir, "vocals")
            
        print(self.target_path)
        
        assert os.path.exists(self.target_path), f"{self.target_path} does not exist!"
        
        for file in os.listdir(self.target_path):
            if file.endswith(".flac"):
                self.file_list.append(file[:-5])
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        key = self.file_list[index]
        file_path = os.path.join(self.target_path, key + ".flac")
        # X, _ = sf.read(file_path, samplerate=self.target_sr)
        try:
            X, _ = librosa.load(file_path, sr=self.target_sr, mono=False)
        except:
            return self.__getitem__(np.random.randint(len(self.file_list)))
        if X.shape[0] > 1:
            # if not mono, take random channel
            channel_id = np.random.randint(X.shape[0])
            X = X[channel_id]
            # X = np.expand_dims(X, axis=-1)
        X_pad = pad_random(X, self.cut)
        X_pad = X_pad / np.max(np.abs(X_pad))
        x_inp = Tensor(X_pad)
        y = int(key.split("_")[0])
        return x_inp, y