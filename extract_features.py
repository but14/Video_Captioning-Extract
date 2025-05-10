import shutil
import numpy as np
import cv2
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import config


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def video_to_frames(video):
    path = os.path.join(config.train_path, 'temporary_images')
    create_directory(path)
    video_path = os.path.join(config.train_path, 'video', video)
    count = 0
    image_list = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Video {video} could not be opened.")
        return []  # Trả về danh sách rỗng nếu video không mở được
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(config.train_path, 'temporary_images', f'frame{count}.jpg')
        cv2.imwrite(frame_path, frame)
        image_list.append(frame_path)
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"{len(image_list)} frames extracted for video {video}.")
    return image_list


def model_cnn_load():
    model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    out = model.layers[-2].output
    model_final = Model(inputs=model.input, outputs=out)
    return model_final


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img


def extract_features(video, model):
    video_id = video.split(".")[0]
    print(f'Processing video {video}')
    image_list = video_to_frames(video)
    samples = np.round(np.linspace(0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]
    images = np.zeros((len(image_list), 224, 224, 3))
    
    for i, img_path in enumerate(image_list):
        img = load_image(img_path)
        images[i] = img
    
    images = np.array(images)
    fc_feats = model.predict(images, batch_size=128)
    img_feats = np.array(fc_feats)
    
    # Cleanup temporary images
    shutil.rmtree(os.path.join(config.train_path, 'temporary_images'))
    return img_feats


def extract_feats_pretrained_cnn():
    model = model_cnn_load()
    print('Model loaded')

    create_directory(os.path.join(config.train_path, 'feat'))

    video_list = [v for v in os.listdir(os.path.join(config.train_path, 'video')) if not v.startswith('.')]
    
    for video in video_list:
        outfile = os.path.join(config.train_path, 'feat', video + '.npy')
        img_feats = extract_features(video, model)
        np.save(outfile, img_feats)
        print(f"Saved features for {video} to {outfile}")


if __name__ == "__main__":
    extract_feats_pretrained_cnn()
