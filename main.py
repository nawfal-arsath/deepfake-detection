import hashlib
import json
import datetime
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import librosa
import torchvision.transforms as transforms
import timm
from moviepy.editor import VideoFileClip
import os
import tempfile
import pickle
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp.isoformat()
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, datetime.datetime.now(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def save_to_disk(self, filename="blockchain.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_disk(cls, filename="blockchain.pkl"):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return cls()

class Node:
    def __init__(self, host, port, blockchain):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.peers = set()

    def start(self):
        # Placeholder for node start logic
        print(f"Node started on {self.host}:{self.port}")

    def broadcast_new_block(self, new_block):
        # Placeholder for broadcasting logic
        print(f"Broadcasting new block: {new_block.hash}")

class ViTDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTDeepfakeDetector, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        num_features = self.vit.head.in_features
        self.vit.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# Load the ViT model
vit_checkpoint = torch.load(r'C:\Users\nawfa\Desktop\deepfake\project\LipFD\models\ckpt.pth', map_location=torch.device('cpu'))
vit_model = ViTDeepfakeDetector()
vit_model.load_state_dict(vit_checkpoint['model'] if 'model' in vit_checkpoint else vit_checkpoint, strict=False)
vit_model.eval()

preprocess_vit = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_spectrogram(audio_file):
    data, sr = librosa.load(audio_file, sr=None)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=sr), ref=np.min)
    return mel

def extract_audio_from_video(video_path, audio_output_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path, codec='pcm_s16le')
    except Exception as e:
        print(f"An error occurred while extracting audio: {e}")

def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_video_with_audio(video_path, audio_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (500, 500))
        frames.append(frame)
        frame_count += 1

    cap.release() 

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30 fps if unable to determine
        print(f"Warning: Unable to determine FPS. Using default value of {fps}")

    duration = frame_count / fps if fps > 0 else 0

    if len(frames) == 0:
        raise ValueError("No frames could be extracted from the video.")

    try:
        mel = get_spectrogram(audio_path)
        mel = cv2.resize(mel, (frames[0].shape[1] * len(frames), frames[0].shape[0]))
        mel = np.expand_dims(mel, axis=2)
        mel = np.repeat(mel, 3, axis=2)
    except Exception as e:
        print(f"Error processing audio: {e}")
        mel = np.zeros((frames[0].shape[0], frames[0].shape[1] * len(frames), 3), dtype=np.uint8)

    combined_images = []
    for i in range(0, len(frames), 5):  # Window size = 5
        window_frames = frames[i:i+5]
        combined_frame = np.concatenate(window_frames, axis=1)
        
        start = int((i / len(frames)) * mel.shape[1])
        end = int(((i + 5) / len(frames)) * mel.shape[1])
        mel_section = mel[:, start:end]
        mel_section = cv2.resize(mel_section, (combined_frame.shape[1], combined_frame.shape[0]))

        combined_image = np.concatenate((mel_section, combined_frame), axis=0)
        combined_images.append(combined_image)
    
    return combined_images, frame_count, fps, duration

def send_email(subject, body, to_email):
    # Email configuration
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "nawfalarsath2005@gmail.com"  # Replace with your email
    sender_password = "yxct xhvl ulwq vgwu" # Replace with your app password

    # Create the email message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def is_video_deepfaked(video_path, blockchain, node):
    try:
        # Extract audio from video
        audio_path = f"{video_path}_audio.wav"
        extract_audio_from_video(video_path, audio_path)
        
        # Preprocess video and audio
        combined_images, frame_count, fps, duration = preprocess_video_with_audio(video_path, audio_path)

        # Perform deepfake detection
        vit_predictions = []
        frame_predictions = []
        
        for i, img in enumerate(combined_images):
            img_tensor = preprocess_vit(img)
            
            with torch.no_grad():
                vit_output = vit_model(img_tensor.unsqueeze(0))
                vit_probabilities = torch.softmax(vit_output, dim=1)
                vit_fake_prob = vit_probabilities[0][1].item() 
                vit_predictions.append(vit_fake_prob)
                frame_predictions.append((i*5, min((i+1)*5, frame_count), vit_fake_prob))

        avg_vit_prediction = sum(vit_predictions) / len(vit_predictions) if vit_predictions else 0

        is_deepfaked = avg_vit_prediction > 0.5
        result = "Fake" if is_deepfaked else "Not Fake"
        
        # Add result to blockchain
        data = {
            "video_path": video_path,
            "result": result,
            "confidence": avg_vit_prediction,
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration,
            "timestamp": datetime.datetime.now().isoformat()
        }
        new_block = Block(len(blockchain.chain), datetime.datetime.now(), data, blockchain.get_latest_block().hash)
        blockchain.add_block(new_block)
        node.broadcast_new_block(new_block)
        
        # Save blockchain to disk after adding new block
        blockchain.save_to_disk()
        
        # Prepare detailed email content
        email_body = f"""
    Deepfake Detection Report

    Video Information:
    - Path: {video_path}
    - Frame Count: {frame_count}
    - FPS: {fps:.2f}
    - Duration: {duration:.2f} seconds

    Detection Result: {result}
    Overall Confidence: {avg_vit_prediction:.2f}

    Frame-by-Frame Analysis:
    """
        for start_frame, end_frame, prob in frame_predictions:
            email_body += f"Frames {start_frame}-{end_frame}: Fake Probability = {prob:.2f}\n"

        email_body += f"\nNote: Probabilities above 0.5 indicate a higher likelihood of being fake."

        # Send email with detailed report
        subject = f"Deepfake Detection Report: {'FAKE DETECTED' if is_deepfaked else 'No Fake Detected'}"
        send_email(subject, email_body, "nawfalarsath2005@gmail.com")  # Replace with the recipient's email
        
        return result, email_body

    except Exception as e:
        error_message = f"An error occurred during video processing: {str(e)}"
        print(error_message)
        return "Error", error_message

def main():
    st.title("Blockchain-Enhanced Deepfake Detection Using ViT")
    
    # Load or create blockchain
    blockchain = Blockchain.load_from_disk()
    
    # Create a node (you may want to make these configurable)
    node = Node("localhost", 5000, blockchain)
    node_thread = threading.Thread(target=node.start)
    node_thread.start()
    
    uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            video_path = temp_file.name
        
        st.video(uploaded_file, format="video/mp4")
        
        with st.spinner("Processing video..."):
            result, email_body = is_video_deepfaked(video_path, blockchain, node)
        
        if result == "Error":
            st.error("An error occurred during video processing.")
            st.text(email_body)  # Display the error message
        else:
            st.write(f"Deepfake Detection Result: {result}")
            st.subheader("Detailed Report")
            st.text(email_body)
            st.success("A detailed email report has been sent.")
        
        # Display blockchain information
        st.subheader("Blockchain Information")
        for block in blockchain.chain:
            st.json(block.__dict__)

if __name__ == "__main__":
    main()