import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import librosa  # Ensure librosa is imported

N_EXTRACT = 10  # Number of extracted images from video
WINDOW_LEN = 5  # Frames of each window

audio_root = "./AVLip/wav"
video_root = "./AVLip"
output_root = "./datasets/val"
temp_root = "./temp"
list_root = "./lists"

labels = [(0, "0_real"), (1, "1_fake")]
max_sample = 100

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from the video and save as a WAV file."""
    try:
        if not os.path.exists(os.path.dirname(audio_path)):
            os.makedirs(os.path.dirname(audio_path))
            
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path, codec='pcm_s16le')  # Ensure using WAV codec
    except Exception as e:
        print(f"Error extracting audio from video: {e}")

def get_spectrogram(audio_file):
    """Generate and save spectrogram from the audio file."""
    try:
        data, sr = librosa.load(audio_file, sr=None)  # Ensure correct sampling rate
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=sr), ref=np.min)
        plt.imsave(f"{temp_root}/mel.png", mel, cmap='inferno')  # Specify colormap
    except Exception as e:
        print(f"Error generating spectrogram: {e}")

def create_list_files():
    """Create the list files if they do not exist."""
    if not os.path.exists(list_root):
        os.makedirs(list_root, exist_ok=True)
    
    fake_list_path = os.path.join(list_root, "fake_list.txt")
    real_list_path = os.path.join(list_root, "real_list.txt")
    
    if not os.path.isfile(fake_list_path):
        with open(fake_list_path, 'w') as f:
            pass
    
    if not os.path.isfile(real_list_path):
        with open(real_list_path, 'w') as f:
            pass

def preprocess_video(video_path, output_dir, group_id, is_fake=False):
    """Preprocess the video by extracting frames and audio."""
    create_list_files()  # Ensure the list files are created
    
    audio_path = f"{temp_root}/audio.wav"
    extract_audio_from_video(video_path, audio_path)
    
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = np.linspace(
        0, frame_count - WINDOW_LEN - 1, N_EXTRACT, endpoint=True, dtype=np.uint8
    ).tolist()
    frame_idx.sort()
    frame_sequence = [i for num in frame_idx for i in range(num, num + WINDOW_LEN)]
    frame_list = []
    current_frame = 0
    while current_frame <= frame_sequence[-1]:
        ret, frame = video_capture.read()
        if not ret:
            print(f"Error reading frame: {current_frame}")
            break
        if current_frame in frame_sequence:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_list.append(cv2.resize(frame, (500, 500)))
        current_frame += 1
    video_capture.release()

    try:
        get_spectrogram(audio_path)
        mel = plt.imread(f"{temp_root}/mel.png") * 255
        mel = mel.astype(np.uint8)
        mapping = mel.shape[1] / frame_count
        for i in range(len(frame_list)):
            idx = i % WINDOW_LEN
            if idx == 0:
                begin = np.round(frame_sequence[i] * mapping)
                end = np.round((frame_sequence[i] + WINDOW_LEN) * mapping)
                sub_mel = cv2.resize(
                    mel[:, int(begin):int(end)], (500 * WINDOW_LEN, 500)
                )
                x = np.concatenate(frame_list[i:i + WINDOW_LEN], axis=1)
                x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

                # Updated image path to ensure unique filenames
                img_path = f"{output_dir}/video_{group_id}_{i}.png"  # Add the index to make each filename unique
                plt.imsave(img_path, x)

                # Debug statement
                list_file = f"{list_root}/fake_list.txt" if is_fake else f"{list_root}/real_list.txt"
                print(f"Appending to {list_file}")

                with open(list_file, 'a') as f:
                    f.write(f"{img_path}\n")
    except Exception as e:
        print(f"Error generating spectrogram: {e}")

    print(f"Finished processing {video_path}.")


def run_preprocessing():
    """Run preprocessing on the dataset."""
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    if not os.path.exists(temp_root):
        os.makedirs(temp_root, exist_ok=True)
    if not os.path.exists(list_root):
        os.makedirs(list_root, exist_ok=True)

    i = 0
    for label, dataset_name in labels:
        if i == max_sample:
            break
        root = f"{video_root}/{dataset_name}"
        video_list = os.listdir(root)
        print(f"Handling {dataset_name}...")
        for j in tqdm(range(len(video_list))):
            v = video_list[j]
            video_path = f"{root}/{v}"
            preprocess_video(video_path, f"{output_root}/{dataset_name}", j, is_fake=(label == 1))
        i += 1

if __name__ == "__main__":
    run_preprocessing()
