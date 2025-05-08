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
from sentence_transformers import SentenceTransformer

# Paths
annotations_file = '/kaggle/input/msvd-dataset-corpus/annotations.txt'
video_clip_dir = '/kaggle/input/msvd-clips/YouTubeClips/'
output_dir = 'outputs/'
os.makedirs(output_dir, exist_ok=True)

# Settings
frame_interval = 30
sharpness_threshold = 100.0

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# User Input
instruction_text = input("Enter instruction (e.g., 'Generate a summarized caption for this video'): ")

# Read Annotations and Create DataFrame
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
    video_paths.append(video_path)
    captions_list.append(captions[0])  # only first caption

df = pd.DataFrame({
    'video_path': video_paths,
    'caption': captions_list
})

print(df.head())
df.to_csv(os.path.join(output_dir, 'video_captions.csv'), index=False)

# Frame Extraction Helpers
def is_sharp(image, threshold=100.0):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    return laplacian_var >= threshold

def extract_frames(video_path, frame_interval, sharpness_threshold):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Failed to open video: {video_path}")
        return frames
    success, image = vidcap.read()
    count = 0
    frame_idx = 0
    while success:
        if count % frame_interval == 0 and is_sharp(image, sharpness_threshold):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            frames.append((frame_idx, pil_image))
        success, image = vidcap.read()
        count += 1
        frame_idx += 1
    vidcap.release()
    return frames

# Dataset
class VideoCaptionDataset(Dataset):
    def __init__(self, dataframe, processor, frame_interval=30, sharpness_threshold=100.0):
        self.data = dataframe
        self.processor = processor
        self.frame_interval = frame_interval
        self.sharpness_threshold = sharpness_threshold
        self.frames_cache = {}
        self._pre_extract_frames()

    def _pre_extract_frames(self):
        print("Pre-extracting frames...")
        for idx in tqdm(range(len(self.data))):
            video_path = self.data.iloc[idx]['video_path']
            if video_path not in self.frames_cache:
                try:
                    frames = extract_frames(video_path, self.frame_interval, self.sharpness_threshold)
                    self.frames_cache[video_path] = frames
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    self.frames_cache[video_path] = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['video_path']
        caption = self.data.iloc[idx]['caption']
        frames = self.frames_cache.get(video_path, [])
        return frames, caption

# Split Data
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.25, random_state=42)

# Create Datasets and Loaders
train_dataset = VideoCaptionDataset(train_df, processor)
val_dataset = VideoCaptionDataset(val_df, processor)
test_dataset = VideoCaptionDataset(test_df, processor)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Loss Function
def calculate_loss(generated_caption, ground_truth_caption, embedder, device):
    gen_embedding = embedder.encode(generated_caption, convert_to_tensor=True).to(device)
    gt_embedding = embedder.encode(ground_truth_caption, convert_to_tensor=True).to(device)
    cos = nn.CosineSimilarity(dim=0)
    similarity = cos(gen_embedding, gt_embedding)
    loss = 1 - similarity
    return loss

# Training Loop
def train_model(model, train_loader, val_loader, optimizer, processor, embedder, device, epochs=2):
    best_model = None
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        for batch in tqdm(train_loader):
            frames_data, ground_truth_caption = batch
            frames_data = frames_data[0]  # Remove batch dimension
            ground_truth_caption = ground_truth_caption[0]

            previous_caption = instruction_text
            captions = []

            for frame_num, frame in frames_data:
                inputs = processor(images=frame, text=previous_caption, return_tensors="pt").to(device)
                output = model.generate(**inputs, max_length=50)
                caption = processor.decode(output[0], skip_special_tokens=True)
                captions.append((frame_num, caption))
                previous_caption = caption

            final_caption = captions[-1][1] if captions else ""

            # Calculate loss using embeddings
            loss = calculate_loss(final_caption, ground_truth_caption, embedder, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss:.4f}")

        # Validation
        val_loss = validate_model(model, val_loader, processor, embedder, device)
        print(f"Validation Loss after epoch {epoch + 1}: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, os.path.join(output_dir, 'best_model.pth'))
            print("Saved the best model so far.")

    return best_model

# Validation Loop
def validate_model(model, val_loader, processor, embedder, device):
    model.eval()
    total_loss = 0.0
    output_file = os.path.join(output_dir, 'validation_captions.txt')

    with torch.no_grad():
        with open(output_file, 'w') as f:
            for batch in tqdm(val_loader):
                frames_data, ground_truth_caption = batch
                frames_data = frames_data[0]
                ground_truth_caption = ground_truth_caption[0]

                previous_caption = instruction_text
                captions = []

                for frame_num, frame in frames_data:
                    inputs = processor(images=frame, text=previous_caption, return_tensors="pt").to(device)
                    output = model.generate(**inputs, max_length=50)
                    caption = processor.decode(output[0], skip_special_tokens=True)
                    captions.append((frame_num, caption))
                    previous_caption = caption

                final_caption = captions[-1][1] if captions else ""
                loss = calculate_loss(final_caption, ground_truth_caption, embedder, device)
                total_loss += loss.item()

                # Save captions
                f.write(f"Ground Truth: {ground_truth_caption}\nGenerated: {final_caption}\n\n")

    avg_loss = total_loss / len(val_loader)
    return avg_loss

# Main
if __name__ == "__main__":
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    train_model(model, train_loader, val_loader, optimizer, processor, embedder, device, epochs=2)