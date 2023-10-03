import os, sys
from tqdm import tqdm
import demucs.separate
from pyannote.audio import Model, Inference

authtoken = None # Fill in your auth token
assert authtoken is not None, "You must provide an auth token to use PyAnnote VAD pipeline."

model = Model.from_pretrained("pyannote/segmentation", use_auth_token=authtoken)

from pyannote.audio.pipelines import VoiceActivityDetection
pipeline = VoiceActivityDetection(segmentation=model)

HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 3.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0,
}
pipeline.instantiate(HYPER_PARAMETERS)

def run_vad(file_path):
    vad = pipeline(file_path)
    vad = str(vad)
    # note that this step only generates a "*.vad" file
    # this file will be of the same name but different extension (.vad)
    with open(file_path.replace(".wav", ".vad"), "w") as f:
        f.write(vad)

if len(sys.argv)!= 2:
    print("Usage: python dataset-separate.py <download_dump_dir>")
    print("The 'download_dump_dir' is the directory where the download dump files are stored.")
    print("Expecting results in the format produced by dataset-download.py")
    exit(1)

download_dump_dir = sys.argv[1]

default_output_dir = "./mdx_extra"

# note that the following are just one of many implementation possibilities

# this part assumes you keep some sort of log during download.
logs = []
for filename in os.listdir(download_dump_dir):
    if filename.endswith(".log"):
        logs.append(os.path.join(download_dump_dir, filename))
        
target_file_names = []
for log in logs:
    # here we parse the logs you kept during download.
    with open(log, "r") as f:
        lines = f.readlines()
        # we assume the first line of your download function is a target file name.
        target_file_names.append(lines[0].strip() + ".flac")

folder_names = os.listdir(default_output_dir)
for name in target_file_names:
    # if appears already in the output folder, skip
    for folder_name in folder_names:
        if folder_name == name.replace(".flac", ""):
            print(name + " appears already in the output folder, skipping...")
            target_file_names.remove(name)

for target_file_name in tqdm(target_file_names, desc="Separate Vocals"):
    # we call demucs to separate the vocals.
    os.system("demucs --two-stems=vocals -n mdx_extra \"" + os.path.join(download_dump_dir, target_file_name) + "\"" )

    # after then, we run vocals.wav through the vad defined as above.
    run_vad(os.path.join(default_output_dir, target_file_name.replace(".flac", "/vocals.wav")))