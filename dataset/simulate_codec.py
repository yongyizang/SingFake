from pydub import AudioSegment
import os, shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import subprocess

# This is the script used for simulating T03 (codec set) from T02.

def simulate_codec(input_path, temp_path, output_path, format, bitrate="128k", codec=None):
    # Convert the input file to the temporary path using ffmpeg
    if codec:
        subprocess.run(['ffmpeg', '-i', input_path, '-acodec', codec, '-b:a', bitrate, '-y', temp_path])
    else:
        subprocess.run(['ffmpeg', '-i', input_path, '-b:a', bitrate, '-y', temp_path])
    # Read the temporary file and convert it back to flac
    subprocess.run(['ffmpeg', '-i', temp_path, '-acodec', 'flac', '-y', output_path])

    # Remove the temporary file
    os.remove(temp_path)

    
def format_suffix(format):
    if format == "mp3":
        return "mp3"
    elif format == "adts":
        return "aac"
    elif format == "ogg":
        return "ogg"
    elif format == "opus":
        return "opus"
    else:
        raise ValueError(f"Invalid format {format}")
    
def format_codec(format):
    if format == "mp3":
        return "libmp3lame"
    elif format == "ogg":
        return "libvorbis"
    elif format == "adts":
        return None
    else:
        # raise ValueError(f"Invalid format {format}")
        return None

def worker(file_tuple):
    src_file_path, dest_file_path, format, bitrate, src_file_name = file_tuple
    temp_path = os.path.join(dest_file_path.split(".flac")[0] + f".{format_suffix(format)}")
    try:
        simulate_codec(src_file_path, temp_path, dest_file_path, format=format, bitrate=bitrate, codec=format_codec(format))
    except Exception as e:
        print(f"Failed to simulate {src_file_path}")
        print(e)

def process_audio_files(src_folder, dest_folder, format="mp3", bitrate="128k", max_workers=None):
    file_list = []
    for subdir, _, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.flac'):
                src_file_path = os.path.join(subdir, file)
                
                # Create the corresponding output directory
                rel_path = os.path.relpath(subdir, src_folder)
                dest_subdir = os.path.join(dest_folder, rel_path)
                os.makedirs(dest_subdir, exist_ok=True)
                
                dest_file_path = os.path.join(dest_subdir, file)
                
                file_list.append((src_file_path, dest_file_path, format, bitrate, file))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(worker, file_list), total=len(file_list), desc=f"Processing {src_folder}"))

audio_formats = None # write audio formats you want
assert audio_formats is not None, "You must specify audio codec formats!"
# example: [["ogg", "64k"], ["opus", "64k"]]

for audio_format_tuple in audio_formats:
    audio_format = audio_format_tuple[0]
    bitrate = audio_format_tuple[1]
    print("Processing " + audio_format + "...")
    src_folder = "/home/yongyi/split_0831/test"
    dest_folder = "/home/yongyi/split_0831/codec_test/"
    dest_folder = dest_folder + audio_format + "_" + bitrate + "/"
    process_audio_files(src_folder, dest_folder, format=audio_format, bitrate=bitrate, max_workers=8)
