import os, sys, librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


# customize these three paths

source_folder = "./mdx_extra" # where the separated files are. This script assumes you have ran separate.py before, and *.vad are under the same folder with file name adhereing to the same logic as written in separate.py.

logs_folder = "./logs" # It needs the *.log files (similar to separate.py)

dump_folder = "./split_dump" # output

output_sr = 16000 # sample rate for output of all files.

vad_files = []

for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".vad"):
            vad_files.append(os.path.join(root, file))

for vad_file in tqdm(vad_files, desc="Rendering Splits"):
    vocal_only = vad_file.replace('.vad', '.wav')
    corresponding_log = os.path.join(logs_folder, vad_file.split("/")[-2].split("_")[0] + ".log")
    # here, it uses the logs file during download to determine if the file is spoof or bonafide.
    spoof = 0
    with open(corresponding_log, "r") as f:
        curr_lines = f.readlines()
        if curr_lines[4].strip() == "spoof":
            spoof = 1
        # this tag is directly written in file name!
        # dataloaders in models/(model) also adhere to this logic.
        mixture_file_name = curr_lines[0].strip() + ".flac"
    mixture_file = os.path.join(logs_folder, mixture_file_name)
    
    # open both
    vocal, vocal_sr = librosa.load(vocal_only, sr=output_sr, mono=False)
    mixture, mixture_sr = librosa.load(mixture_file, sr=output_sr, mono=False)
    
    isMono = False
    
    if len(vocal.shape) == 1:
        isMono = True
    
    if len(mixture.shape) == 1:
        isMono = True
    
    index = 0
    
    with open(vad_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            else:
                try:
                    items = line.split("]")[0].split("[")[1].split("-->")
                    start_times = items[0].split(":")
                    end_times = items[1].split(":")
                    
                    start_time = float(start_times[2]) + float(start_times[1]) * 60.0 + float(start_times[0]) * 3600.0
                    end_time = float(end_times[2]) + float(end_times[1]) * 60.0 + float(end_times[0]) * 3600.0
                    if isMono:
                        vocal_curr_seg = vocal[int(start_time * output_sr):int(end_time * output_sr)]
                        mixture_curr_seg = mixture[int(start_time * output_sr):int(end_time * output_sr)]
                        vocal_curr_seg = np.expand_dims(vocal_curr_seg, axis=0)
                        mixture_curr_seg = np.expand_dims(mixture_curr_seg, axis=0)
                        
                        # transpose
                        vocal_curr_seg = np.transpose(vocal_curr_seg)
                        mixture_curr_seg = np.transpose(mixture_curr_seg)
                    else:
                        vocal_curr_seg = vocal[:, int(start_time * output_sr):int(end_time * output_sr)]
                        mixture_curr_seg = mixture[:, int(start_time * output_sr):int(end_time * output_sr)]
                        
                        # transpose
                        vocal_curr_seg = np.transpose(vocal_curr_seg)
                        mixture_curr_seg = np.transpose(mixture_curr_seg)
                    
                    file_name = str(spoof) + "_" + vad_file.split("/")[-2].split("_")[0] + "_" + str(index) + ".flac"
                    
                    # save vocal
                    sf.write(os.path.join(dump_folder, "vocals", file_name), vocal_curr_seg, vocal_sr, subtype="PCM_16")
                    
                    # save mixture
                    sf.write(os.path.join(dump_folder, "mixtures", file_name), mixture_curr_seg, mixture_sr, subtype="PCM_16")
                    
                    index += 1
                except Exception as e:
                    print(e)
                    continue