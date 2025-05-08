# train_video_captioning.py

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image, sequence, text
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Masking
from sklearn.model_selection import train_test_split
import tensorflow as tf

# --- Config ---
FRAME_STEP = 10
MAX_FRAMES = 10
FEATURE_DIM = 512
MAX_LEN = 20
EMBEDDING_DIM = 256
LSTM_UNITS = 256

# --- Paths ---
VIDEO_DIR = "videos"
CAPTION_CSV = "captions.csv"
FRAME_DIR = "frames"

# --- Load captions ---
df = pd.read_csv(CAPTION_CSV)
df = df.groupby("video_id").first().reset_index()  # chỉ lấy 1 caption đầu tiên

# --- Tokenizer ---
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(df['caption'])
VOCAB_SIZE = len(tokenizer.word_index) + 1

def extract_frames(video_path, output_folder, every_n=FRAME_STEP):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()
    i = 0
    while success:
        if i % every_n == 0:
            filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            frame_count += 1
            if frame_count >= MAX_FRAMES:
                break
        success, frame = cap.read()
        i += 1
    cap.release()

def extract_video_features(video_path, vgg_model):
    temp_dir = os.path.join(FRAME_DIR, os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(temp_dir, exist_ok=True)
    extract_frames(video_path, temp_dir)
    features = []
    for frame_file in sorted(os.listdir(temp_dir))[:MAX_FRAMES]:
        img_path = os.path.join(temp_dir, frame_file)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = vgg_model.predict(x, verbose=0).squeeze()
        features.append(feat)
    while len(features) < MAX_FRAMES:
        features.append(np.zeros(FEATURE_DIM))
    return np.array(features)

# --- VGG16 model ---
vgg = VGG16(weights='imagenet', include_top=False, pooling='avg')

X = []
Y = []

print("Extracting video features...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    video_path = os.path.join(VIDEO_DIR, row['video_id'])
    caption = row['caption']
    features = extract_video_features(video_path, vgg)
    seq = tokenizer.texts_to_sequences([caption])[0]
    seq = sequence.pad_sequences([seq], maxlen=MAX_LEN, padding='post')[0]
    one_hot = tf.keras.utils.to_categorical(seq, num_classes=VOCAB_SIZE)
    X.append(features)
    Y.append(one_hot)

X = np.array(X)
Y = np.array(Y)

# --- Model ---
input_video = Input(shape=(MAX_FRAMES, FEATURE_DIM))
x = Masking()(input_video)
x = LSTM(LSTM_UNITS)(x)
x = Dense(VOCAB_SIZE, activation='softmax')(x)

model = Model(inputs=input_video, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Train ---
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=8)

# --- Save ---
model.save("video_caption_model.h5")
with open("tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())
