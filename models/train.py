import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import cv2
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

# Paths
annotations_file = '/kaggle/input/msvd-dataset-corpus/annotations.txt'
video_clip_dir = '/kaggle/input/msvd-clips/YouTubeClips/'
output_dir = 'outputs/frames/'

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Settings
frame_interval = 30
sharpness_threshold = 100.0

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Read annotations and create DataFrame with all captions
video_data = {}
with open(annotations_file, 'r') as file:
    for line in file:
        clip_id, caption = line.strip().split(' ', 1)
        if clip_id not in video_data:
            video_data[clip_id] = []
        video_data[clip_id].append(caption)

video_paths = []
captions_list = []
for video_id, captions in video_data.items():
    video_path = f"{video_clip_dir}{video_id}.avi"
    for caption in captions:  # Use all captions
        video_paths.append(video_path)
        captions_list.append(caption)

df = pd.DataFrame({
    'video_path': video_paths,
    'caption': captions_list
})

print(df.head())
df.to_csv('video_captions.csv', index=False)

# Frame extraction helpers
def is_sharp(image, threshold=100.0):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    return laplacian_var >= threshold

def preprocess_frames(video_path, frame_interval, sharpness_threshold, output_dir):
    video_id = os.path.basename(video_path).split('.')[0]
    frame_dir = os.path.join(output_dir, video_id)
    os.makedirs(frame_dir, exist_ok=True)
    frames = []
    try:
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return frames
        success, image = vidcap.read()
        count = 0
        frame_idx = 0
        while success:
            if count % frame_interval == 0 and is_sharp(image, sharpness_threshold):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                frame_path = os.path.join(frame_dir, f"frame_{frame_idx}.png")
                pil_image.save(frame_path)
                frames.append((frame_idx, frame_path))
            success, image = vidcap.read()
            count += 1
            frame_idx += 1
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    finally:
        vidcap.release()
    return frames

# Preprocess all videos
for video_path in df['video_path'].unique():
    preprocess_frames(video_path, frame_interval, sharpness_threshold, output_dir)

# Dataset
class VideoCaptionDataset(Dataset):
    def __init__(self, dataframe, processor, output_dir):
        self.data = dataframe
        self.processor = processor
        self.output_dir = output_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['video_path']
        caption = self.data.iloc[idx]['caption']
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return [], caption
        video_id = os.path.basename(video_path).split('.')[0]
        frame_dir = os.path.join(self.output_dir, video_id)
        frames = []
        if os.path.exists(frame_dir):
            for frame_file in sorted(os.listdir(frame_dir)):
                frame_path = os.path.join(frame_dir, frame_file)
                frame_idx = int(frame_file.split('_')[1].split('.')[0])
                pil_image = Image.open(frame_path)
                frames.append((frame_idx, pil_image))
        return frames, caption

# Split Data
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.25, random_state=42)

# Create Datasets and Loaders
train_dataset = VideoCaptionDataset(train_df, processor, output_dir)
val_dataset = VideoCaptionDataset(val_df, processor, output_dir)
test_dataset = VideoCaptionDataset(test_df, processor, output_dir)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Training Loop
def train_model(model, optimizer, train_loader, processor, device, epochs=5):
    best_model = None
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{epochs}")

        for batch in tqdm(train_loader):
            frames_data, ground_truth_caption = batch
            ground_truth_caption = ground_truth_caption[0]

            if not frames_data or not frames_data[0]:  # Check if frames are empty
                print("Skipping: No valid frames")
                continue

            frames_data = frames_data[0]  # Remove batch dimension
            captions = []

            for idx, (frame_num, frame) in enumerate(frames_data):
                inputs = processor(images=frame, text=ground_truth_caption, return_tensors="pt").to(device)
                output = model.generate(**inputs, max_length=50)
                caption = processor.decode(output[0], skip_special_tokens=True)
                captions.append((frame_num, caption))

            final_caption = captions[-1][1] if captions else ""

            # Calculate loss using BLIP's CrossEntropyLoss
            inputs = processor(images=frames_data[-1][1], text=ground_truth_caption, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss:.4f}")

        # Validation
        val_loss = validate_model(model, val_loader, processor, device)
        print(f"Validation Loss after epoch {epoch + 1}: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, 'best_model.pth')
            print("Saved the best model so far.")

    return best_model

# Validation Loop
def validate_model(model, val_loader, processor, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            frames_data, ground_truth_caption = batch
            ground_truth_caption = ground_truth_caption[0]

            if not frames_data or not frames_data[0]:
                print("Skipping: No valid frames")
                continue

            frames_data = frames_data[0]
            captions = []

            for idx, (frame_num, frame) in enumerate(frames_data):
                inputs = processor(images=frame, text=ground_truth_caption, return_tensors="pt").to(device)
                output = model.generate(**inputs, max_length=50)
                caption = processor.decode(output[0], skip_special_tokens=True)
                captions.append((frame_num, caption))

            final_caption = captions[-1][1] if captions else ""

            # Calculate loss
            inputs = processor(images=frames_data[-1][1], text=ground_truth_caption, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Train and Evaluate
best_model = train_model(model, optimizer, train_loader, processor, device, epochs=2)
evaluate_model(model, test_loader, processor, device)
